## Inspiration
Reinforcement Learning (RL) has achieved significant success in decision-making tasks but struggles with real-world applications where interaction with the environment is expensive or dangerous. Offline RL attempts to address this by learning from fixed datasets, but suffers from Q-value overestimation on out-of-distribution (OOD) state-action pairs.

The Conformal Q-Learning approach seeks to mitigate these issues by integrating conformal prediction into RL. This method provides distribution-free uncertainty quantification with finite-sample guarantees, helping to stabilize policy learning and prevent overestimation errors. The approach is particularly relevant for safety-critical applications where robust decision-making is necessary.

## What it does
Conformal Q-Learning enhances standard RL algorithms by introducing statistical confidence intervals around Q-value estimates. It ensures that:
- Learned Q-values remain within prediction intervals with high probability.
- Uncertainty quantification helps mitigate overestimation and unsafe policy decisions.
- Policies become more stable and robust to OOD state-action pairs.
- It improves conservatism and optimism balance, performing better than traditional Conservative Q-Learning (CQL) approaches.

## How we built it

The Conformal Q-Learning framework was developed by integrating conformal prediction into an actor-critic RL setup, specifically for offline RL. The methodology includes:

- Q-Network Training: A deep Q-network is trained to estimate Q-values based on historical data.
- Conformal Interval Calibration: During training, nonconformity scores are computed using a calibration set to construct prediction intervals.
- Actor-Critic Framework: The actor network (policy) is updated to maximize Q-values, incorporating the uncertainty information from conformal intervals.
- Empirical evaluations were conducted on CartPole-v1 using offline RL datasets, validating the effectiveness of Conformal Q-Learning.

## Challenges we ran into
1. Handling OOD Actions – Offline RL suffers from extrapolation errors, and ensuring robust Q-value estimates in unseen states was non-trivial.
2. Balancing Conservatism & Optimism – Unlike CQL, which applies a fixed penalty, conformal prediction needed fine-tuned quantile selection.
3. Computational Constraints – Ensuring that conformal interval calibration remains computationally feasible without excessive overhead.
4. Stability in Training – Traditional RL algorithms can suffer from instability, particularly when using confidence intervals in decision-making.

## Accomplishments that we're proud of
- Empirical Success: Demonstrated improved policy stability, robustness to OOD data, and enhanced performance compared to CQL and standard DQN.
- First-of-its-kind integration of conformal prediction into Q-learning for uncertainty quantification in RL.

## What we learned
- Uncertainty estimation is critical in RL, particularly for offline settings where exploration is limited.
- Accelerating RL training with PyTorch optimizations: Leveraging techniques such as torch.jit for just-in-time (JIT) compilation, CUDA acceleration, and efficient tensor operations can significantly speed up Q-network training and conformal interval calculations, making the approach more scalable for real-world applications.

## What's next for Stable-RL
We are looking to expand the amount of algorithms that we can integrate this in. Currently, we had time to implement this idea with Soft-Actor Critic, but integrating this uncertainty quantification into more popular algorithms is our next goal. Additionally testing its robustness on edge-based scenarios, like autonomous driving, is a plausible next step as our motivation was centered around improving robustness in these areas
