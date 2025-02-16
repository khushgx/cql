# run_agent.py

from conformal_agent.agent_wrapper import SACAgent

def main():
    agent = SACAgent(
        env_name="halfcheetah-medium-expert",
        offline=True,
        iteration=10,  #
        seed=42,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        log_interval=5,
        alpha_q=100,
        q_alpha_update_freq=50
    )

    agent.train()

    score = agent.evaluate(eval_episodes=2)
    print(f"Final evaluation score: {score}")

if __name__ == "__main__":
    main()
