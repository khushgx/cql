"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function OfflineLearningPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Offline Learning</h1>
            <p className="text-lg text-white/70">
              Understanding the offline reinforcement learning paradigm in RL-CP Fusion
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
              Offline learning is a crucial component of RL-CP Fusion, allowing the algorithm to learn effective
              policies from static datasets without direct environment interaction.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">What is Offline Learning?</h2>
          <p className="mb-6">
            Offline learning, also known as batch reinforcement learning, is an approach where an agent learns a policy
            from a fixed dataset of experiences without interacting with the environment during training. This paradigm
            is particularly useful in scenarios where direct interaction with the environment is impractical, expensive,
            or dangerous.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Key Concepts</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Static Dataset</h3>
          <p className="mb-6">
            In offline RL, the agent learns from a pre-collected dataset of transitions (state, action, reward, next
            state). This dataset is typically gathered using some behavior policy or a combination of policies.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Distributional Shift</h3>
          <p className="mb-6">
            One of the main challenges in offline RL is distributional shift, where the learned policy may encounter
            states or actions that are not well-represented in the static dataset.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Extrapolation Error</h3>
          <p className="mb-6">
            Extrapolation error occurs when the Q-function produces unreliable, often overly optimistic estimates for
            out-of-distribution (OOD) state-action pairs.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Offline Learning in RL-CP Fusion</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Integration with Conformal Prediction</h3>
          <p className="mb-6">
            RL-CP Fusion addresses the challenges of offline learning by incorporating conformal prediction to provide
            reliable uncertainty estimates for Q-values, particularly for OOD actions.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Conservative Q-Learning</h3>
          <p className="mb-6">
            The algorithm builds upon the principles of Conservative Q-Learning (CQL) but uses adaptive, data-driven
            penalties based on conformal intervals instead of fixed conservative penalties.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Policy Constraint</h3>
          <p className="mb-6">
            RL-CP Fusion constrains the learned policy to actions that have low uncertainty according to the conformal
            prediction intervals, helping to mitigate the effects of distributional shift.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Advantages of Offline Learning in RL-CP Fusion</h2>
          <ul className="list-disc list-inside mb-6">
            <li>Enables learning from historical data without the need for online interaction</li>
            <li>Reduces the risk associated with exploring in sensitive or dangerous environments</li>
            <li>Allows for more efficient use of data, especially in scenarios where data collection is expensive</li>
            <li>
              Facilitates the use of large, diverse datasets that may be impractical to collect in online settings
            </li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Challenges and Solutions</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Limited Exploration</h3>
          <p className="mb-6">
            Challenge: The agent cannot actively explore to gather new information. Solution: RL-CP Fusion uses
            conformal prediction to quantify uncertainty and guide the policy towards actions with reliable estimates.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Overestimation Bias</h3>
          <p className="mb-6">
            Challenge: Q-function tends to overestimate values for unseen state-action pairs. Solution: Conformal
            intervals provide adaptive regularization, penalizing uncertain estimates more heavily.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Policy Constraint</h3>
          <p className="mb-6">
            Challenge: Ensuring the learned policy stays close to the data distribution. Solution: The algorithm
            incorporates the width of conformal intervals into the policy update, naturally constraining it to regions
            with low uncertainty.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Experimental Results</h2>
          <p className="mb-6">
            Experiments on standard RL benchmarks demonstrate that RL-CP Fusion's offline learning approach:
          </p>
          <ul className="list-disc list-inside mb-6">
            <li>Achieves comparable or better performance than online methods in many tasks</li>
            <li>Shows improved robustness to out-of-distribution actions compared to baseline offline RL methods</li>
            <li>Provides reliable uncertainty estimates, leading to more stable and safe policies</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Conclusion</h2>
          <p className="mb-6">
            Offline learning is a critical component of RL-CP Fusion, enabling the algorithm to learn effective policies
            from static datasets while addressing key challenges such as distributional shift and extrapolation error.
            By combining offline learning with conformal prediction, RL-CP Fusion offers a robust and theoretically
            grounded approach to reinforcement learning in scenarios where online interaction is limited or infeasible.
          </p>

          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mt-8">
            <p className="text-sm text-white/70">
              Next, explore the{" "}
              <a href="/docs/api/sacagent" className="text-blue-400 hover:text-blue-300">
                API Reference
              </a>
              to learn how to implement RL-CP Fusion in your projects.
            </p>
          </div>
        </div>
      </div>
    </DocLayout>
  )
}

