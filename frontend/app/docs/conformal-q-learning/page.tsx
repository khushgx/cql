"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function ConformalQLearningPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Conformal Q-Learning</h1>
            <p className="text-lg text-white/70">Understanding the core component of RL-CP Fusion</p>
          </div>
          <Button variant="ghost" className="text-white/70 hover:text-white">
            <Copy className="mr-2 h-4 w-4" />
            Copy page
          </Button>
        </div>

        <div className="prose prose-invert max-w-none">
          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mb-8">
            <p className="text-sm text-white/70">
              Conformal Q-Learning integrates conformal prediction into an actor-critic framework to address
              extrapolation error in offline reinforcement learning.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">What is Conformal Q-Learning?</h2>
          <p className="mb-6">
            Conformal Q-Learning is a novel approach that combines conformal prediction with actor-critic methods to
            provide finite-sample uncertainty guarantees for Q-value estimates in offline reinforcement learning. It
            constructs prediction intervals around learned Q-values to ensure that true values lie within these
            intervals with high probability, using interval width as a regularizer to mitigate overestimation and
            stabilize policy learning.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Key Components</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Conformal Prediction Integration</h3>
          <p className="mb-6">
            Conformal prediction is used to generate uncertainty estimates for Q-values. This is crucial in offline RL
            to avoid overconfident estimates for out-of-distribution actions. The algorithm maintains a calibration set
            and periodically updates conformal intervals to ensure reliable uncertainty quantification.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Actor-Critic Framework</h3>
          <p className="mb-6">
            Conformal Q-Learning builds upon the actor-critic architecture, where a critic (Q-function) estimates
            action-values and an actor (policy) selects actions. The conformal intervals are incorporated into both the
            critic updates and policy improvement steps.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Offline Learning Mechanism</h3>
          <p className="mb-6">
            The algorithm is designed to learn from static datasets without interacting with the environment during
            training. It uses techniques like conformal regularization to mitigate the challenges of offline RL, such as
            extrapolation error and distributional shift.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">How Conformal Q-Learning Works</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Initialization</h3>
          <p className="mb-6">
            The algorithm initializes a Q-network, policy network, and a calibration dataset. It also sets up learning
            rates and thresholds for conformal prediction.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Training Loop</h3>
          <p className="mb-6">During training, Conformal Q-Learning:</p>
          <ul className="list-disc list-inside mb-6">
            <li>Samples batches from the offline dataset</li>
            <li>Calibrates conformal intervals using the calibration set</li>
            <li>Updates the Q-network (critic) using the Bellman equation and conformal regularization</li>
            <li>Updates the policy network (actor) incorporating the conformal intervals</li>
          </ul>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Conformal Interval Calibration</h3>
          <p className="mb-6">
            The algorithm computes nonconformity scores and determines the conformal threshold (q_α) based on the
            calibration dataset. This threshold is used to construct prediction intervals for Q-values.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Advantages of Conformal Q-Learning</h2>
          <ul className="list-disc list-inside mb-6">
            <li>Provides uncertainty quantification for Q-value estimates</li>
            <li>Mitigates overestimation bias in offline reinforcement learning</li>
            <li>Improves stability and reliability of learned policies</li>
            <li>Offers theoretical guarantees on the coverage of prediction intervals</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Conclusion</h2>
          <p className="mb-6">
            Conformal Q-Learning represents a significant advancement in offline reinforcement learning by providing
            theoretically grounded uncertainty estimates. By integrating conformal prediction into the actor-critic
            framework, it addresses key challenges in offline RL, such as extrapolation error and policy stability. The
            method's success in both theoretical analysis and empirical evaluations makes it a promising approach for
            reliable offline RL in safety-critical and resource-constrained environments.
          </p>

          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mt-8">
            <p className="text-sm text-white/70">
              Next, learn about{" "}
              <a href="/docs/conformal-prediction" className="text-blue-400 hover:text-blue-300">
                Conformal Prediction
              </a>
              and how it enhances the stability of offline RL in RL-CP Fusion.
            </p>
          </div>
        </div>
      </div>
    </DocLayout>
  )
}

