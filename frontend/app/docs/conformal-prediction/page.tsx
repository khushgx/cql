"use client"

import { Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import DocLayout from "@/components/doc-layout"

export default function ConformalPredictionPage() {
  return (
    <DocLayout>
      <div className="max-w-4xl">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">Conformal Prediction</h1>
            <p className="text-lg text-white/70">
              Understanding the uncertainty quantification technique in RL-CP Fusion
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
              Conformal prediction is a statistical technique used in RL-CP Fusion to provide distribution-free
              uncertainty estimates for Q-value predictions.
            </p>
          </div>

          <h2 className="text-2xl font-bold mt-12 mb-4">What is Conformal Prediction?</h2>
          <p className="mb-6">
            Conformal prediction is a method for constructing prediction intervals with strong statistical guarantees.
            In the context of Conformal Q-Learning, it is used to generate robust confidence intervals for Q-value
            estimates, particularly for optimistic predictions. This helps mitigate the risk of overestimating
            out-of-distribution (OOD) actions and ensures a more stable learning process.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Key Concepts</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Nonconformity Scores</h3>
          <p className="mb-6">
            Nonconformity scores measure how different a new prediction is from observed data. In Conformal Q-Learning,
            these scores are defined as the absolute difference between predicted and observed Q-values:
          </p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">α = |f(s,a) - y_i|</pre>
          <p className="mt-4">Where f(s,a) is the predicted Q-value and y_i is the observed Q-value.</p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Calibration Set</h3>
          <p className="mb-6">
            A subset of the data used to calibrate the conformal prediction intervals. In Conformal Q-Learning, this is
            typically a portion of the offline dataset.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Conformal Intervals</h3>
          <p className="mb-6">
            Prediction intervals constructed using the calibration set and nonconformity scores. For a given confidence
            level 1-α, the interval is defined as:
          </p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">C(s,a) = [f(s,a) - q_α, f(s,a) + q_α]</pre>
          <p className="mt-4">Where q_α is the (1-α)-quantile of the nonconformity scores in the calibration set.</p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Conformal Prediction in RL-CP Fusion</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">1. Integration with Q-Learning</h3>
          <p className="mb-6">
            Conformal prediction is integrated into the Q-learning process by using the constructed intervals to
            regularize Q-value updates and guide policy improvement.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">2. Adaptive Uncertainty Estimation</h3>
          <p className="mb-6">
            The conformal intervals adapt to the underlying data distribution, providing tighter bounds in regions with
            more data and wider bounds in uncertain areas.
          </p>

          <h3 className="text-xl font-semibold mt-8 mb-4">3. Group-Conditional Coverage</h3>
          <p className="mb-6">
            RL-CP Fusion extends conformal prediction to group-conditional coverage, allowing for more fine-grained
            uncertainty estimates across different subsets of the state-action space.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Theoretical Guarantees</h2>

          <h3 className="text-xl font-semibold mt-8 mb-4">Marginal Coverage</h3>
          <p className="mb-6">
            Conformal prediction provides a theoretical guarantee on marginal coverage. For a new test point (s_m+1,
            a_m+1, y_m+1), the probability that y_m+1 lies within the conformal interval C(s_m+1, a_m+1) is bounded:
          </p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">P(y_m+1 ∈ C(s_m+1, a_m+1)) ∈ [1-α, 1-α+1/(m+1)]</pre>

          <h3 className="text-xl font-semibold mt-8 mb-4">Group-Conditional Coverage</h3>
          <p className="mb-6">For group-conditional coverage, the prediction intervals satisfy:</p>
          <pre className="bg-gray-800 text-white p-4 rounded-md">1-α-ε_j ≤ P(y ∈ C(s,a) | (s,a) ∈ G_j) ≤ 1-α+ε_j</pre>
          <p className="mt-4">
            Where G_j are groups defined over the state-action space, and ε_j is an error term that vanishes as the
            calibration data size within each group grows.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-4">Benefits in Offline RL</h2>
          <ul className="list-disc list-inside mb-6">
            <li>Mitigates overestimation bias for out-of-distribution actions</li>
            <li>Provides reliable uncertainty quantification without modifying the underlying RL architecture</li>
            <li>Enables more stable and conservative policy learning in offline settings</li>
            <li>Offers theoretical guarantees on the reliability of Q-value estimates</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-4">Conclusion</h2>
          <p className="mb-6">
            Conformal prediction plays a crucial role in RL-CP Fusion by providing theoretically grounded uncertainty
            estimates for Q-values. This integration enhances the stability and reliability of offline reinforcement
            learning, making it particularly valuable for applications in safety-critical and resource-constrained
            environments.
          </p>

          <div className="rounded-lg bg-white/5 border border-white/10 p-4 mt-8">
            <p className="text-sm text-white/70">
              Next, learn about{" "}
              <a href="/docs/offline-learning" className="text-blue-400 hover:text-blue-300">
                Offline Learning
              </a>
              and how it complements Conformal Prediction in RL-CP Fusion.
            </p>
          </div>
        </div>
      </div>
    </DocLayout>
  )
}

