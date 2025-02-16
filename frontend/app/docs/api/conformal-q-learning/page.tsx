"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function ConformalQLearningAPIPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Conformal Q-Learning</h1>
            <p className="text-lg text-white/70">API reference for the main Conformal Q-Learning implementation</p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">
              The SACAgent class implements Conformal Q-Learning, combining Soft Actor-Critic (SAC) with Conformal
              Prediction for offline reinforcement learning.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">SACAgent Class</h2>
          <p className="mb-4 text-white/70">
            The SACAgent class is a high-level wrapper for the Conformal SAC agent, providing an easy-to-use interface
            for training and evaluation.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">Constructor</h3>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`class SACAgent:
    def __init__(self, env_name: str, offline: bool = True, iteration: int = 100000, seed: int = 1, **config):
        # Initialize the Conformal Q-Learning agent
        ...`}
          </pre>

          <h4 className="text-lg font-semibold mt-6 mb-2">Parameters</h4>
          <ul className="space-y-4">
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">env_name</code>
              <span className="text-white/70 ml-2">string</span>
              <p className="mt-1 text-white/70">Name of the Gym (or D4RL) environment.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">offline</code>
              <span className="text-white/70 ml-2">boolean, default: True</span>
              <p className="mt-1 text-white/70">If True, use an offline dataset from D4RL.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">iteration</code>
              <span className="text-white/70 ml-2">integer, default: 100000</span>
              <p className="mt-1 text-white/70">Number of training iterations.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">seed</code>
              <span className="text-white/70 ml-2">integer, default: 1</span>
              <p className="mt-1 text-white/70">Random seed for reproducibility.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">**config</code>
              <span className="text-white/70 ml-2">dict</span>
              <p className="mt-1 text-white/70">Additional hyperparameters for the underlying SAC agent.</p>
            </li>
          </ul>

          <h3 className="text-xl font-semibold mt-8 mb-4">Methods</h3>

          <h4 className="text-lg font-semibold mt-6 mb-2">train()</h4>
          <p className="mb-4 text-white/70">Runs the main training loop for the Conformal Q-Learning algorithm.</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`def train(self):
    # Main training loop
    ...`}
          </pre>

          <h4 className="text-lg font-semibold mt-6 mb-2">evaluate(eval_episodes: int = 5) → float</h4>
          <p className="mb-4 text-white/70">Evaluates the current policy on the environment.</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`def evaluate(self, eval_episodes: int = 5) -> float:
    # Evaluate the policy
    ...`}
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-4">SAC Class</h2>
          <p className="mb-4 text-white/70">
            The SAC class implements the core Soft Actor-Critic algorithm with Conformal Prediction integration.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">Constructor</h3>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`class SAC:
    def __init__(self, state_dim, action_dim, config):
        # Initialize the SAC agent
        ...`}
          </pre>

          <h4 className="text-lg font-semibold mt-6 mb-2">Key Methods</h4>
          <ul className="space-y-4">
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">select_action(state)</code>
              <p className="mt-1 text-white/70">Selects an action given a state.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">store(s, a, r, s_, d)</code>
              <p className="mt-1 text-white/70">Stores a transition in the replay buffer.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">update()</code>
              <p className="mt-1 text-white/70">Performs one update step using sampled transitions.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">compute_conformal_interval(alpha=0.1)</code>
              <p className="mt-1 text-white/70">Computes the conformal interval for uncertainty estimation.</p>
            </li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Example Usage</h2>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`from conformal_sac.agent_wrapper import SACAgent

agent = SACAgent(
    env_name="halfcheetah-medium-expert",
    offline=True,
    iteration=100000,
    seed=42,
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    log_interval=2000,
    alpha_q=100,
    q_alpha_update_freq=50
)

agent.train()
score = agent.evaluate(eval_episodes=5)
print(f"Final evaluation score: {score}")
`}
          </pre>

          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mt-8">
            <p className="text-sm text-white/70">
              For more details on configuration options, see the{" "}
              <a href="/docs/api/configuration" className="text-blue-400 hover:text-blue-300">
                Configuration
              </a>{" "}
              page.
            </p>
          </div>
        </div>
      </div>
    </DocLayout>
  )
}

