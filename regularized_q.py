import argparse
from collections import namedtuple
import os
import numpy as np
import random
import gym
import d4rl  # Make sure to install d4rl: pip install d4rl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

'''
Implementation of Soft Actor-Critic, dual Q network version (Offline Mode using D4RL)
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation!
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument("--env_name", default="hopper-medium")  # A D4RL environment, e.g., hopper-medium-v2
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)

# Use float for learning_rate and gamma
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=float)  # discount gamma
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size (set large for offline datasets)
parser.add_argument('--iteration', default=100000, type=int)  # num of training iterations
parser.add_argument('--batch_size', default=256, type=int)  # mini batch size
parser.add_argument('--seed', default=1, type=int)

# Optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=2000, type=int)
parser.add_argument('--load', default=False, type=bool)  # load model

# New argument for offline training using D4RL
parser.add_argument('--offline', default=True, type=bool, help="Train offline using a D4RL dataset.")

args = parser.parse_args()

# Define a Transition tuple
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])

# Offline training uses D4RL so we need to create an environment for dataset loading.
env = gym.make(args.env_name)

# Seed for reproducibility
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Get dimensions from the dataset:
dataset = d4rl.qlearning_dataset(env)
if "next_observations" in dataset:
    next_obs = dataset["next_observations"]
else:
    next_obs = np.concatenate([dataset["observations"][1:], dataset["observations"][-1:]], axis=0)

state_dim = dataset["observations"].shape[1]
action_dim = dataset["actions"].shape[1]
# Assume actions are normalized between -1 and 1.
max_action = 1.0

min_Val = torch.tensor(1e-7, dtype=torch.float32, device=device)

# --- Network Definitions ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # Update: output dimension should match action_dim.
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        # Update: log_std now has shape (batch, action_dim)
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
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        x = torch.cat((s, a), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- SAC Agent Definition ---
class SAC():
    def __init__(self):
        self.policy_net = Actor(state_dim, action_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        # Initialize replay buffer
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

        # 🔹 Step 1: Create a Calibration Set for Conformal Prediction
        self.calibration_ratio = 0.1  # 10% of dataset for calibration
        self.calibration_set = []
        self.q_alpha = 0.0
        self.alpha_q = 0.1 #regularization strength
        self.q_alpha_update_freq = 100  # Update q_alpha every N steps


    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        # Update: sample noise with the same shape as mu.
        z = Normal(torch.zeros_like(mu), torch.ones_like(sigma)).sample()
        action = torch.tanh(mu + sigma * z).detach().cpu().numpy()
        return action  # now returns a vector of size action_dim

    def store(self, s, a, r, s_, d):
        self.replay_buffer.append(Transition(s, a, r, s_, d))
        self.num_transition += 1

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = Normal(torch.zeros_like(batch_mu), torch.ones_like(batch_sigma)).sample().to(device)
        action = torch.tanh(batch_mu + batch_sigma * z)
        log_prob = dist.log_prob(batch_mu + batch_sigma * z) - torch.log(1 - action.pow(2) + min_Val)
        # Sum log probabilities over action dimensions to get a scalar per sample.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update_calibration_set(self):
        """ Dynamically updates the calibration set to avoid memory overhead. """
        calibration_size = int(len(self.replay_buffer) * self.calibration_ratio)
        indices = np.random.choice(len(self.replay_buffer), calibration_size, replace=False)
        self.calibration_set = [self.replay_buffer[i] for i in indices]

    def compute_conformal_interval(self, alpha=0.1):
        """Compute the conformal interval with reduced computation frequency."""
        if len(self.calibration_set) == 0:
            return 0.0  # Return zero uncertainty if no calibration data

        with torch.no_grad():
            # Reduce calibration size dynamically to speed up processing
            max_calibration_samples = 1000  # Avoid processing too many samples
            sample_indices = np.random.choice(len(self.calibration_set), 
                                            min(len(self.calibration_set), max_calibration_samples), 
                                            replace=False)
            sampled_calibration = [self.calibration_set[i] for i in sample_indices]

            # Convert to batch tensors
            bn_s = torch.tensor(np.array([t.s for t in sampled_calibration]), dtype=torch.float32, device=device)
            bn_a = torch.tensor(np.array([t.a for t in sampled_calibration]), dtype=torch.float32, device=device)
            bn_r = torch.tensor(np.array([t.r for t in sampled_calibration]), dtype=torch.float32, device=device).view(-1, 1)
            bn_s_ = torch.tensor(np.array([t.s_ for t in sampled_calibration]), dtype=torch.float32, device=device)
            bn_d = torch.tensor(np.array([t.d for t in sampled_calibration]), dtype=torch.float32, device=device).view(-1, 1)

            # Compute Q-values in batch
            q_values = self.Q_net1(bn_s, bn_a).squeeze()
            next_values = self.Target_value_net(bn_s_).squeeze()

            # Compute residuals
            y_values = bn_r + (1 - bn_d) * args.gamma * next_values
            residuals = torch.abs(q_values - y_values)

            # Compute quantile for uncertainty estimation
            q_alpha = torch.quantile(residuals, 1 - alpha).item()

        return q_alpha
    
    def compute_conformal_loss(self, bn_s, bn_a):
        """ Computes Q-value regularization using the conformal interval q_alpha. """
        with torch.no_grad():
            random_actions = torch.FloatTensor(bn_s.shape[0], action_dim).uniform_(-1, 1).to(device)
            next_actions, _, _, _, _ = self.evaluate(bn_s)

        q_random = self.Q_net1(bn_s, random_actions)
        q_next = self.Q_net1(bn_s, next_actions)
        q_dataset = self.Q_net1(bn_s, bn_a)

        # Add conformal penalty to the CQL-style regularization
        conformal_penalty = self.q_alpha / (torch.abs(q_dataset).mean() + 1e-3)
        
        cql_loss = torch.logsumexp(q_random, dim=0) - q_dataset.mean() - conformal_penalty
        return self.alpha_q * cql_loss


    def update(self):
        if self.num_training % 500 == 0:
            print(f"Training ... {self.num_training} times.")

        for _ in range(args.gradient_steps):
            # Sample a batch of transitions from the replay buffer
            indices = np.random.choice(len(self.replay_buffer), args.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in indices]

            # Convert only the sampled batch to tensors
            bn_s = torch.tensor(np.array([t.s for t in batch]), dtype=torch.float32, device=device)
            bn_a = torch.tensor(np.array([t.a for t in batch]), dtype=torch.float32, device=device).view(-1, action_dim)
            bn_r = torch.tensor(np.array([t.r for t in batch]), dtype=torch.float32, device=device).view(-1, 1)
            bn_s_ = torch.tensor(np.array([t.s_ for t in batch]), dtype=torch.float32, device=device)
            bn_d = torch.tensor(np.array([t.d for t in batch]), dtype=torch.float32, device=device).view(-1, 1)


            # 1. Compute target Q-value
            with torch.no_grad():
                target_value_next = self.Target_value_net(bn_s_)
                next_q_value = bn_r + (1 - bn_d) * args.gamma * target_value_next

            # 2. Compute current estimates from value and Q networks
            current_value = self.value_net(bn_s)
            current_Q1 = self.Q_net1(bn_s, bn_a)
            current_Q2 = self.Q_net2(bn_s, bn_a)

            conformal_loss1 = self.compute_conformal_loss(bn_s, bn_a)
            conformal_loss2 = self.compute_conformal_loss(bn_s, bn_a)

            # 3. Evaluate policy for on-policy Q-value estimation
            sample_action, log_prob, _, _, _ = self.evaluate(bn_s)
            Q1_pi = self.Q_net1(bn_s, sample_action)
            Q2_pi = self.Q_net2(bn_s, sample_action)
            current_Q_pi = torch.min(Q1_pi, Q2_pi)

            # 4. Next value for computing the value loss
            next_value = current_Q_pi - log_prob


            # Adjust Q-value update using conformal interval width
            q_alpha_scaled = self.q_alpha / (torch.std(next_value).mean() + 1e-3)  # Normalize q_alpha
            next_value = next_value * (1 - q_alpha_scaled)

            
            if self.num_training % self.q_alpha_update_freq == 0:
                self.update_calibration_set()
                self.q_alpha = self.compute_conformal_interval(alpha=0.1)
                print(f'Recalibrated q_alpha {self.q_alpha}')


            # 5. Compute losses for each component
            V_loss = self.value_criterion(current_value, next_value.detach()).mean()
            Q1_loss = self.Q1_criterion(current_Q1, next_q_value.detach()).mean() + conformal_loss1
            Q2_loss = self.Q2_criterion(current_Q2, next_q_value.detach()).mean() + conformal_loss2
            pi_loss = (log_prob - current_Q_pi).mean()

            # 6. Backpropagate combined loss
            self.value_optimizer.zero_grad()
            self.Q1_optimizer.zero_grad()
            self.Q2_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()

            total_loss = V_loss + Q1_loss + Q2_loss + pi_loss
            total_loss.backward()

            # Clip gradients for stability
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)

            self.value_optimizer.step()
            self.Q1_optimizer.step()
            self.Q2_optimizer.step()
            self.policy_optimizer.step()

            # 7. Soft-update the target network
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - args.tau) + param * args.tau)

            # 8. Logging losses
            self.writer.add_scalar('Loss/V_loss', V_loss.item(), self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss.item(), self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss.item(), self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss.item(), self.num_training)
            self.writer.add_scalar('Uncertainty Update', self.q_alpha, self.num_training)
            self.num_training += 1


    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load('./SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        print("Model has been loaded.")

# --- Main Training Loop ---
def evaluate_policy(env, agent, eval_episodes=5):
    """Runs evaluation episodes using the current policy and returns the average normalized reward."""
    avg_reward = 0.0
    uncertainties = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(np.float32(action))
            episode_reward += reward

            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).to(device).unsqueeze(0)
            q_value = agent.Q_net1(state_tensor, action_tensor).item()
            q_alpha = agent.compute_conformal_interval(alpha=0.1)
            uncertainties.append(q_alpha)
            
        avg_reward += episode_reward
    avg_reward /= eval_episodes
    avg_uncertainty = np.mean(uncertainties)

    
    # Compute normalized return using D4RL's method
    normalized_score = d4rl.get_normalized_score(env.spec.id, avg_reward) * 100

    print(f"Evaluation - Avg Reward: {avg_reward:.2f}, Normalized Return: {normalized_score:.2f}, Avg Uncertainty: {avg_uncertainty:.2f}")
    agent.writer.add_scalar('eval/normalized_return', normalized_score, agent.num_training)
    agent.writer.add_scalar('eval/uncertainty', avg_uncertainty, agent.num_training)
    return normalized_score

def main():
    agent = SAC()
    if args.load:
        agent.load()

    if args.offline:
        # Load offline dataset using D4RL
        print("Loading offline dataset using D4RL...")
        dataset = d4rl.qlearning_dataset(env)
        if "next_observations" in dataset:
            next_obs = dataset["next_observations"]
        else:
            next_obs = np.concatenate([dataset["observations"][1:], dataset["observations"][-1:]], axis=0)

        agent.replay_buffer = []
        num_transitions = dataset["observations"].shape[0]
        for i in range(num_transitions):
            s = dataset["observations"][i]
            a = dataset["actions"][i]
            r = dataset["rewards"][i]
            s_ = next_obs[i]
            d = dataset["terminals"][i]
            agent.replay_buffer.append(Transition(s, a, r, s_, d))
        agent.num_transition = len(agent.replay_buffer)
        print("Offline dataset loaded, total transitions:", agent.num_transition)
        calibration_size = int(len(agent.replay_buffer) * agent.calibration_ratio)
        agent.calibration_set = random.sample(agent.replay_buffer, calibration_size)
        # Offline training loop: update repeatedly using the fixed dataset.
        best_normalized_return = -np.inf
        early_stop_patience = 3  # Stop training if no improvement for N evaluations
        no_improvement_steps = 0

        for i in range(args.iteration):
            agent.update()
            
            if i % args.log_interval == 0:
                # Compute normalized return
                normalized_return = evaluate_policy(env, agent, eval_episodes=5)
                agent.writer.add_scalar('eval/normalized_return', normalized_return, i)

                # Save the best model
                if normalized_return > best_normalized_return:
                    best_normalized_return = normalized_return
                    agent.save()
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1

                # # Stop training if no improvement for a while
                # if no_improvement_steps >= early_stop_patience:
                #     print(f"Early stopping at iteration {i} due to no improvement in normalized return.")
                #     break
                        
        print("Offline training complete.")

    else:
        # Online training (if needed)
        if args.render:
            env.render()
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        ep_r = 0
        for i in range(args.iteration):
            state = env.reset()
            for t in range(200):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                if args.render:
                    env.render()
                agent.store(state, action, reward, next_state, done)
                if agent.num_transition >= args.capacity:
                    agent.update()
                state = next_state
                if done or t == 199:
                    if i % 10 == 0:
                        print(f"Ep_i {i}, Episode Reward: {ep_r}, Time Steps: {t}")
                    break
            if i % args.log_interval == 0:
                agent.save()
            agent.writer.add_scalar('ep_r', ep_r, global_step=i)
            ep_r = 0

if __name__ == '__main__':
    main()

