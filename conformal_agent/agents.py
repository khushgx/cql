# File: conformal_sac/agent.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import namedtuple
from tensorboardX import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])

max_action = 1.0
min_Val = torch.tensor(1e-7, dtype=torch.float32, device=device)

# --- Network Definitions ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = F.relu(self.log_std_head(x))
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        # s and a are expected to be batch tensors.
        x = torch.cat((s, a), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- SAC Agent Definition ---
class SAC:
    def __init__(self, state_dim, action_dim, config):
        """
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            config (dict): Hyperparameters and configuration options. For example:
                {
                  'learning_rate': 3e-4,
                  'gamma': 0.99,
                  'tau': 0.005,
                  'batch_size': 256,
                  'gradient_steps': 1,
                  'alpha_q': 100,
                  'q_alpha_update_freq': 50,
                  ...
                }
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = Actor(state_dim, action_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.get('learning_rate', 3e-4))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.get('learning_rate', 3e-4))
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=config.get('learning_rate', 3e-4))
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=config.get('learning_rate', 3e-4))

        self.replay_buffer = []
        self.num_transition = 0
        self.num_training = 1

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        self.writer = SummaryWriter('./exp-SAC_dual_Q_network')
        os.makedirs('./SAC_model/', exist_ok=True)

        # Sync target network with value network.
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # Conformal prediction calibration variables
        self.calibration_ratio = 0.1  # 10% of the replay buffer for calibration
        self.calibration_set = []
        self.q_alpha = 0.0
        self.alpha_q = config.get('alpha_q', 100)  # regularization strength
        self.q_alpha_update_freq = config.get('q_alpha_update_freq', 50)  # update frequency for q_alpha
        self.q_target_ema = 0.0

    def select_action(self, state):
        """Select an action given a state."""
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        z = Normal(torch.zeros_like(mu), torch.ones_like(sigma)).sample()
        action = torch.tanh(mu + sigma * z).detach().cpu().numpy()
        return action

    def store(self, s, a, r, s_, d):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append(Transition(s, a, r, s_, d))
        self.num_transition += 1

    def evaluate(self, state):
        """
        Evaluate the policy network.
        Args:
            state (torch.Tensor): A batch of states.
        Returns:
            action, log_prob, noise, mu, log_sigma
        """
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        z = Normal(torch.zeros_like(batch_mu), torch.ones_like(batch_sigma)).sample().to(device)
        action = torch.tanh(batch_mu + batch_sigma * z)
        log_prob = Normal(batch_mu, batch_sigma).log_prob(batch_mu + batch_sigma * z) - torch.log(1 - action.pow(2) + min_Val)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update_calibration_set(self):
        """Dynamically update the calibration set from the replay buffer."""
        calibration_size = int(len(self.replay_buffer) * self.calibration_ratio)
        indices = np.random.choice(len(self.replay_buffer), calibration_size, replace=False)
        self.calibration_set = [self.replay_buffer[i] for i in indices]

    def compute_conformal_interval(self, alpha=0.1):
        """
        Compute the conformal interval (q_alpha) for uncertainty estimation.
        Args:
            alpha (float): The quantile level.
        Returns:
            q_alpha (float): The computed uncertainty estimate.
        """
        if len(self.calibration_set) == 0:
            return 0.0
        with torch.no_grad():
            max_samples = 1000
            sample_indices = np.random.choice(len(self.calibration_set),
                                              min(len(self.calibration_set), max_samples),
                                              replace=False)
            sampled = [self.calibration_set[i] for i in sample_indices]
            bn_s = torch.tensor(np.array([t.s for t in sampled]), dtype=torch.float32, device=device)
            bn_a = torch.tensor(np.array([t.a for t in sampled]), dtype=torch.float32, device=device)
            bn_r = torch.tensor(np.array([t.r for t in sampled]), dtype=torch.float32, device=device).view(-1, 1)
            bn_s_ = torch.tensor(np.array([t.s_ for t in sampled]), dtype=torch.float32, device=device)
            bn_d = torch.tensor(np.array([t.d for t in sampled]), dtype=torch.float32, device=device).view(-1, 1)
            q_values = self.Q_net1(bn_s, bn_a).squeeze()
            gamma = self.config.get('gamma', 0.99)
            y_values = bn_r + (1 - bn_d) * gamma * self.Target_value_net(bn_s_).squeeze()
            self.q_target_ema = 0.95 * self.q_target_ema + 0.05 * y_values.mean().item()
            residuals = torch.abs(q_values - self.q_target_ema)
            q_alpha = torch.quantile(residuals, 1 - alpha).item()
        return q_alpha

    def compute_conformal_loss(self, bn_s, bn_a):
        """
        Compute the conformal loss component for Q-value regularization.
        Args:
            bn_s (torch.Tensor): Batch of states.
            bn_a (torch.Tensor): Batch of actions.
        Returns:
            A conformal loss term (torch.Tensor).
        """
        with torch.no_grad():
            random_actions = torch.FloatTensor(bn_s.shape[0], self.action_dim).uniform_(-1, 1).to(device)
            next_actions, _, _, _, _ = self.evaluate(bn_s)
        q_random = self.Q_net1(bn_s, random_actions)
        q_dataset = self.Q_net1(bn_s, bn_a)
        conformal_penalty = torch.clamp(self.q_alpha / (torch.abs(q_dataset).mean() + 1e-3), max=5.0)
        cql_loss = torch.logsumexp(q_random, dim=0) - q_dataset.mean() - conformal_penalty
        return self.alpha_q * cql_loss

    def update(self):
        """
        Perform one update step (or several gradient steps) using transitions sampled from the replay buffer.
        This method includes:
          1. Sampling a mini-batch.
          2. Computing target Q-values.
          3. Computing the losses for the value network, Q networks, and policy.
          4. Backpropagation and optimizer steps.
          5. Soft-updating the target network.
          6. Logging the training progress.
        """
        if self.num_training % 500 == 0:
            print(f"Training ... {self.num_training} times.")

        gradient_steps = self.config.get('gradient_steps', 1)
        batch_size = self.config.get('batch_size', 256)
        gamma = self.config.get('gamma', 0.99)
        tau = self.config.get('tau', 0.005)

        for _ in range(gradient_steps):
            # Sample a mini-batch from the replay buffer
            indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in indices]

            # Convert the mini-batch to tensors
            bn_s = torch.tensor(np.array([t.s for t in batch]), dtype=torch.float32, device=device)
            bn_a = torch.tensor(np.array([t.a for t in batch]), dtype=torch.float32, device=device).view(-1, self.action_dim)
            bn_r = torch.tensor(np.array([t.r for t in batch]), dtype=torch.float32, device=device).view(-1, 1)
            bn_s_ = torch.tensor(np.array([t.s_ for t in batch]), dtype=torch.float32, device=device)
            bn_d = torch.tensor(np.array([t.d for t in batch]), dtype=torch.float32, device=device).view(-1, 1)

            # 1. Compute target Q-value
            with torch.no_grad():
                target_value_next = self.Target_value_net(bn_s_)
                next_q_value = bn_r + (1 - bn_d) * gamma * target_value_next

            # 2. Compute current estimates from the value and Q networks
            current_value = self.value_net(bn_s)
            current_Q1 = self.Q_net1(bn_s, bn_a)
            current_Q2 = self.Q_net2(bn_s, bn_a)

            conformal_loss1 = self.compute_conformal_loss(bn_s, bn_a)
            conformal_loss2 = self.compute_conformal_loss(bn_s, bn_a)

            # 3. Evaluate the policy for on-policy Q-value estimation
            sample_action, log_prob, _, _, _ = self.evaluate(bn_s)
            Q1_pi = self.Q_net1(bn_s, sample_action)
            Q2_pi = self.Q_net2(bn_s, sample_action)
            current_Q_pi = torch.min(Q1_pi, Q2_pi)

            # 4. Compute the next value for the value loss
            next_value = current_Q_pi - log_prob
            q_alpha_scaled = self.q_alpha / (torch.std(next_value).mean() + 1e-3)
            next_value = next_value * (1 - q_alpha_scaled)

            # Update calibration and q_alpha at specified frequency
            if self.num_training % self.q_alpha_update_freq == 0:
                self.update_calibration_set()
                new_q_alpha = self.compute_conformal_interval(alpha=0.1)
                self.q_alpha = 0.95 * self.q_alpha + 0.05 * new_q_alpha
                print(f"Recalibrated q_alpha: {self.q_alpha}")

            # 5. Compute losses for each component
            V_loss = self.value_criterion(current_value, next_value.detach()).mean()
            Q1_loss = self.Q1_criterion(current_Q1, next_q_value.detach()).mean() + conformal_loss1
            Q2_loss = self.Q2_criterion(current_Q2, next_q_value.detach()).mean() + conformal_loss2
            pi_loss = (log_prob - current_Q_pi).mean()

            # 6. Backpropagation and optimizer steps
            self.value_optimizer.zero_grad()
            self.Q1_optimizer.zero_grad()
            self.Q2_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()

            total_loss = V_loss + Q1_loss + Q2_loss + pi_loss
            total_loss.backward()

            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)

            self.value_optimizer.step()
            self.Q1_optimizer.step()
            self.Q2_optimizer.step()
            self.policy_optimizer.step()

            # 7. Soft-update the target value network
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - tau) + param * tau)

            # 8. Log training progress
            self.writer.add_scalar('Loss/V_loss', V_loss.item(), self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss.item(), self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss.item(), self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss.item(), self.num_training)
            self.writer.add_scalar('Uncertainty Update', self.q_alpha, self.num_training)
            self.num_training += 1

    def save(self):
        """Save the model weights."""
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        print("Model saved.")

    def load(self):
        """Load the model weights."""
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load('./SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        print("Model loaded.")
