from __future__ import annotations

import glob
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from edge_viot_env import EdgeVIoTEnv


def train(env, steps: int = 10_000, seed: int | None = 0):
    env.reset(seed=seed)

    print("Starting training.")

    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"viot_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print("Finished training.")
    
    env.close()


def eval(env, model_path, num_games: int = 10):

    print("Starting evaluation")

    model = PPO.load(model_path)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    env = EdgeVIoTEnv(render_mode="ansi")

    train(env, steps=81_920, seed=0)

    # eval(env, num_games=10, render_mode=None, **env_kwargs)

    # eval(env, num_games=2, render_mode="human", **env_kwargs)