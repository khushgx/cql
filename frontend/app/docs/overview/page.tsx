"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function OverviewPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Overview</h1>
            <p className="text-lg text-white/70">Introduction to RL-CP Fusion and its core concepts</p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">
              RL-CP Fusion combines Offline Reinforcement Learning with Conformal Prediction to create robust and
              reliable learning algorithms with uncertainty quantification.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">Background</h2>
          <p className="mb-6">
            Reinforcement Learning (RL) has emerged as a versatile paradigm in the field of machine learning, capable of
            tackling complex decision-making problems where agents learn to optimize long-term rewards through
            interactions with an environment. From language model alignment to video game playing, RL has demonstrated
            immense success, especially in environments that enjoy no penalty for exploratory action.
          </p>

          <p className="mb-6">
            Unfortunately, real-world applications often present constraints where direct interaction with the
            environment is either impractical, prohibitively expensive, or downright dangerous. In the absence of
            simulators, Offline or Batch RL attempts to sidestep this by garnering large datasets upstream of train time
            and applying slightly adjusted versions of the Online techniques -- albeit to varying degrees of success.
            The catch: no dynamic data collection or exploration during training. The offline setting requires learning
            effective policies solely from a static dataset -- presenting a vast set of new challenges.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">The Challenge of Offline RL</h2>
          <p className="mb-6">
            A common strategy in data-constrained settings is bootstrapping. However, employing standard value-based
            off-policy RL algorithms with bootstrapping frequently leads to poor performance. This approach tends to
            produce overly optimistic value function estimates, particularly for sparse or out-of-distribution (OOD)
            actions, known as extrapolation error. To address this, previous work has incorporated complex
            regularization terms into model objectives, aiming to establish lower bounds on both Q-function estimates
            and policy performance.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Our Approach: RL-CP Fusion</h2>
          <p className="mb-6">
            Our project takes a different approach by integrating Conformal Prediction (CP)—a statistical tool for
            quantifying uncertainty—to enhance the stability and reliability of Offline RL. By leveraging CP, we aim to
            generate robust confidence intervals for Q-value estimates, particularly for optimistic predictions. This
            helps mitigate the risk of overestimating OOD actions and ensures a more stable learning process, while also
            providing meaningful coverage guarantees.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Key Features of RL-CP Fusion</h2>
          <ul className="space-y-4">
            <li className="flex items-start">
              <span className="bg-white/10 p-1 rounded mr-3 mt-1">
                <svg className="h-4 w-4 text-white/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </span>
              <div>
                <h3 className="text-lg font-semibold">Offline Learning</h3>
                <p className="text-white/70">
                  Learn effective policies from static datasets without direct environment interaction.
                </p>
              </div>
            </li>
            <li className="flex items-start">
              <span className="bg-white/10 p-1 rounded mr-3 mt-1">
                <svg className="h-4 w-4 text-white/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </span>
              <div>
                <h3 className="text-lg font-semibold">Conformal Prediction Integration</h3>
                <p className="text-white/70">
                  Leverage Conformal Prediction for robust confidence intervals on Q-value estimates.
                </p>
              </div>
            </li>
            <li className="flex items-start">
              <span className="bg-white/10 p-1 rounded mr-3 mt-1">
                <svg className="h-4 w-4 text-white/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </span>
              <div>
                <h3 className="text-lg font-semibold">Stable & Reliable Learning</h3>
                <p className="text-white/70">
                  Mitigate overestimation risks and ensure a more stable learning process.
                </p>
              </div>
            </li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Getting Started</h2>
          <p className="mb-6">To get started with RL-CP Fusion, follow these steps:</p>
          <ol className="space-y-4 list-decimal list-inside">
            <li className="text-white/70">
              Follow the{" "}
              <a href="/docs/quickstart" className="text-blue-400 hover:text-blue-300">
                Quickstart guide
              </a>{" "}
              to train your first agent
            </li>
            <li className="text-white/70">
              Learn about the core concepts in the{" "}
              <a href="/docs/sacagent" className="text-blue-400 hover:text-blue-300">
                SACAgent documentation
              </a>
            </li>
            <li className="text-white/70">
              Explore the{" "}
              <a href="/docs/api/sacagent" className="text-blue-400 hover:text-blue-300">
                API Reference
              </a>{" "}
              for detailed information
            </li>
          </ol>
        </div>
      </div>
    </DocLayout>
  )
}

