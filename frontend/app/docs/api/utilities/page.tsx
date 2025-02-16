"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function UtilitiesAPIPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Utilities</h1>
            <p className="text-lg text-white/70">
              API reference for utility functions and classes in Conformal Q-Learning
            </p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">
              This page provides details on utility functions and helper classes used in the Conformal Q-Learning
              implementation.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Replay Buffer</h2>
          <p className="mb-4 text-white/70">
            The Replay Buffer is used to store and sample experiences for offline learning.
          </p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    
    def store(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
`}
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-4">Neural Network Models</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">Actor Network</h3>
          <p className="mb-4 text-white/70">The Actor network is used to learn the policy in the SAC algorithm.</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = F.relu(self.log_std_head(x))
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std
`}
          </pre>

          <h3 className="text-xl font-semibold mt-8 mb-4">Critic Network</h3>
          <p className="mb-4 text-white/70">The Critic network is used to estimate the Q-value function.</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
`}
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-4">Conformal Prediction Utilities</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">compute_conformal_interval</h3>
          <p className="mb-4 text-white/70">Computes the conformal interval for uncertainty estimation.</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`def compute_conformal_interval(self, alpha=0.1):
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
`}
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-4">Evaluation Utilities</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">evaluate</h3>
          <p className="mb-4 text-white/70">Evaluates the current policy on the environment.</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`def evaluate(self, eval_episodes: int = 5) -> float:
    total_reward = 0.0
    for _ in range(eval_episodes):
        state = self.env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = self.agent.select_action(state)
            state, reward, done, _ = self.env.step(action.astype(np.float32))
            ep_reward += reward
        total_reward += ep_reward
    avg_reward = total_reward / eval_episodes
    return avg_reward
`}
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-4">Logging Utilities</h2>

          <p className="mb-4 text-white/70">
            The implementation uses TensorBoardX for logging training progress and results.
          </p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`from tensorboardX import SummaryWriter

self.writer = SummaryWriter('./exp-SAC_dual_Q_network')

# Example usage in the update method
self.writer.add_scalar('Loss/V_loss', V_loss.item(), self.num_training)
self.writer.add_scalar('Loss/Q1_loss', Q1_loss.item(), self.num_training)
self.writer.add_scalar('Loss/Q2_loss', Q2_loss.item(), self.num_training)
self.writer.add_scalar('Loss/policy_loss', pi_loss.item(), self.num_training)
self.writer.add_scalar('Uncertainty Update', self.q_alpha, self.num_training)
`}
          </pre>

          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mt-8">
            <p className="text-sm text-white/70">
              For more information on the core algorithm, see the{" "}
              <a href="/docs/api/conformal-q-learning" className="text-blue-400 hover:text-blue-300">
                Conformal Q-Learning
              </a>{" "}
              page.
            </p>
          </div>
        </div>
      </div>
    </DocLayout>
  )
}

