"use client"
import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function QuickstartPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Quickstart</h1>
            <p className="text-lg text-white/70">Get started with RL-CP Fusion in minutes</p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">
              This guide will help you train your first agent using RL-CP Fusion with a D4RL dataset.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Prerequisites</h2>
          <p className="mb-4">Before you begin, make sure you have:</p>
          <ul className="space-y-2 list-disc list-inside mb-8">
            <li className="text-white/70">Python 3.7 or higher installed</li>
            <li className="text-white/70">pip package manager</li>
            <li className="text-white/70">Basic understanding of reinforcement learning concepts</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Installation</h2>
          <p className="mb-4">First, install the required packages:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`pip install torch gym d4rl numpy tensorboardX`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Training Your First Agent</h2>
          <p className="mb-4">Here's a complete example to train an agent on the HalfCheetah environment:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`from conformal_sac.agent_wrapper import SACAgent

# Initialize the agent
agent = SACAgent(
    env_name="halfcheetah-medium-v2",
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

# Train the agent
agent.train()

# Evaluate the trained agent
score = agent.evaluate(eval_episodes=10)
print(f"Final evaluation score: {score}")`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Understanding the Code</h2>
          <p className="mb-4">Let's break down what's happening in the code above:</p>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Agent Initialization</h3>
          <p className="mb-4">We create a new SACAgent instance with specific hyperparameters:</p>
          <ul className="space-y-2 list-disc list-inside mb-6">
            <li className="text-white/70">
              <code>env_name</code>: The D4RL environment to use
            </li>
            <li className="text-white/70">
              <code>offline</code>: Set to True for offline learning
            </li>
            <li className="text-white/70">
              <code>iteration</code>: Number of training iterations
            </li>
            <li className="text-white/70">
              <code>seed</code>: Random seed for reproducibility
            </li>
            <li className="text-white/70">Various hyperparameters for the SAC algorithm</li>
          </ul>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Training</h3>
          <p className="mb-4">
            The <code>train()</code> method:
          </p>
          <ul className="space-y-2 list-disc list-inside mb-6">
            <li className="text-white/70">Loads the offline dataset</li>
            <li className="text-white/70">Performs training iterations</li>
            <li className="text-white/70">Updates the agent's policy</li>
            <li className="text-white/70">Periodically evaluates performance</li>
          </ul>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Evaluation</h3>
          <p className="mb-4">
            The <code>evaluate()</code> method runs the trained policy for multiple episodes and returns the average
            reward.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Monitoring Training</h2>
          <p className="mb-4">You can monitor the training progress using TensorBoard:</p>
          <div className="bg-[#0a0a0a] rounded-lg p-4 mb-6">
            <pre className="text-sm text-white/90">
              <code>{`tensorboard --logdir ./exp-SAC_dual_Q_network`}</code>
            </pre>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Next Steps</h2>
          <p className="mb-4">Now that you've trained your first agent, you can:</p>
          <ul className="space-y-2 list-disc list-inside">
            <li className="text-white/70">
              Learn about{" "}
              <a href="/docs/conformal-prediction" className="text-blue-400 hover:text-blue-300">
                Conformal Prediction
              </a>
            </li>
            <li className="text-white/70">
              Understand{" "}
              <a href="/docs/offline-learning" className="text-blue-400 hover:text-blue-300">
                Offline Learning
              </a>
            </li>
            <li className="text-white/70">
              Explore the{" "}
              <a href="/docs/api/sacagent" className="text-blue-400 hover:text-blue-300">
                API Reference
              </a>
            </li>
          </ul>
        </div>
      </div>
    </DocLayout>
  )
}

