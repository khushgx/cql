# File: conformal_sac/agent_wrapper.py

import gym
import d4rl  # Make sure d4rl is installed: pip install d4rl
import numpy as np
import torch
import random
from .agents import SAC  # Import the SAC agent from your agent.py

class SACAgent:
    """
    A high-level wrapper for the Conformal SAC agent.
    
    Example usage:
    
        from conformal_sac.agent_wrapper import SACAgent
        
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
    """
    def __init__(self, env_name: str, offline: bool = True, iteration: int = 100000, seed: int = 1, **config):
        """
        Initializes the high-level SACAgent.
        
        Args:
            env_name (str): Name of the Gym (or D4RL) environment.
            offline (bool): If True, use an offline dataset from D4RL.
            iteration (int): Number of training iterations.
            seed (int): Random seed for reproducibility.
            **config: Additional hyperparameters (learning_rate, gamma, tau, batch_size, etc.).
        """
        self.env_name = env_name
        self.offline = offline
        self.iteration = iteration
        self.seed = seed
        self.config = config  # Hyperparameters for the underlying SAC agent

        # Create and seed the environment
        self.env = gym.make(env_name)
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # If offline training, use D4RL to load a dataset and get state/action dimensions.
        if offline:
            self.dataset = d4rl.qlearning_dataset(self.env)
            self.state_dim = self.dataset["observations"].shape[1]
            self.action_dim = self.dataset["actions"].shape[1]
        else:
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]

        # Instantiate the underlying SAC agent
        self.agent = SAC(self.state_dim, self.action_dim, self.config)

        # If offline, load the dataset into the agent's replay buffer
        if offline:
            if "next_observations" in self.dataset:
                next_obs = self.dataset["next_observations"]
            else:
                next_obs = np.concatenate(
                    [self.dataset["observations"][1:], self.dataset["observations"][-1:]], axis=0
                )
            num_transitions = self.dataset["observations"].shape[0]
            for i in range(num_transitions):
                s = self.dataset["observations"][i]
                a = self.dataset["actions"][i]
                r = self.dataset["rewards"][i]
                s_ = next_obs[i]
                d = self.dataset["terminals"][i]
                self.agent.store(s, a, r, s_, d)
            print(f"Offline dataset loaded with {num_transitions} transitions.")

    def train(self):
        """
        Runs the training loop.
        During training, the agent's update method is called repeatedly.
        Evaluation is performed every log_interval steps.
        """
        best_score = -float("inf")
        log_interval = self.config.get("log_interval", 2000)
        
        for i in range(self.iteration):
            self.agent.update()
            
            if i % log_interval == 0:
                score = self.evaluate(eval_episodes=5)
                print(f"Iteration {i}: Eval Score = {score}")
                
                # Save the model if improved
                if score > best_score:
                    best_score = score
                    self.agent.save()
        print("Training complete.")

    def evaluate(self, eval_episodes: int = 5) -> float:
        """
        Evaluates the current policy on the environment.
        
        Args:
            eval_episodes (int): Number of episodes to average over.
        
        Returns:
            float: Average reward over the evaluation episodes.
        """
        total_reward = 0.0
        for _ in range(eval_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = self.agent.select_action(state)
                # Ensure the action is in the correct type (e.g., np.float32)
                state, reward, done, _ = self.env.step(action.astype(np.float32))
                ep_reward += reward
            total_reward += ep_reward
        avg_reward = total_reward / eval_episodes
        return avg_reward
