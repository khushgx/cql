"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function ConfigurationAPIPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Configuration</h1>
            <p className="text-lg text-white/70">API reference for configuring Conformal Q-Learning</p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">
              This page provides details on the configuration options available for the Conformal Q-Learning algorithm.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Configuration Options</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">SACAgent Parameters</h3>
          <ul className="space-y-4">
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">env_name</code>
              <span className="text-white/70 ml-2">string</span>
              <p className="mt-1 text-white/70">
                Name of the Gym (or D4RL) environment. Example: "halfcheetah-medium-expert"
              </p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">offline</code>
              <span className="text-white/70 ml-2">boolean, default: True</span>
              <p className="mt-1 text-white/70">
                If True, use an offline dataset from D4RL. If False, interact with the environment during training.
              </p>
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
          </ul>

          <h3 className="text-xl font-semibold mt-8 mb-4">SAC Hyperparameters</h3>
          <ul className="space-y-4">
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">learning_rate</code>
              <span className="text-white/70 ml-2">float, default: 3e-4</span>
              <p className="mt-1 text-white/70">Learning rate for the optimizer.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">gamma</code>
              <span className="text-white/70 ml-2">float, default: 0.99</span>
              <p className="mt-1 text-white/70">Discount factor for future rewards.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">tau</code>
              <span className="text-white/70 ml-2">float, default: 0.005</span>
              <p className="mt-1 text-white/70">Soft update coefficient for target networks.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">batch_size</code>
              <span className="text-white/70 ml-2">integer, default: 256</span>
              <p className="mt-1 text-white/70">Batch size for training.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">log_interval</code>
              <span className="text-white/70 ml-2">integer, default: 2000</span>
              <p className="mt-1 text-white/70">Interval for logging and evaluation.</p>
            </li>
          </ul>

          <h3 className="text-xl font-semibold mt-8 mb-4">Conformal Prediction Parameters</h3>
          <ul className="space-y-4">
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">alpha_q</code>
              <span className="text-white/70 ml-2">float, default: 100</span>
              <p className="mt-1 text-white/70">Coefficient for the conformal regularization term.</p>
            </li>
            <li>
              <code className="bg-white/10 px-2 py-1 rounded text-sm">q_alpha_update_freq</code>
              <span className="text-white/70 ml-2">integer, default: 50</span>
              <p className="mt-1 text-white/70">Frequency of updating the conformal threshold.</p>
            </li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Example Configuration</h2>
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
`}
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-4">Advanced Configuration</h2>
          <p className="mb-4 text-white/70">
            For more advanced use cases, you can create a configuration dictionary and pass it to the SACAgent
            constructor:
          </p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">
            {`config = {
    "env_name": "halfcheetah-medium-expert",
    "offline": True,
    "iteration": 100000,
    "seed": 42,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "log_interval": 2000,
    "alpha_q": 100,
    "q_alpha_update_freq": 50,
    "hidden_sizes": [256, 256],  # Custom neural network architecture
    "activation": "relu",        # Activation function for hidden layers
    "optimizer": "adam"          # Optimizer type
}

agent = SACAgent(**config)
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

