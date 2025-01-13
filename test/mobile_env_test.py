import gymnasium
import mobile_env

env = gymnasium.make("mobile-medium-central-v0")
obs, info = env.reset()
done = False
act_sample = env.action_space.sample()
obs_sample = env.observation_space.sample()

while not done:
    action = env.action_space.sample()  # Your agent code here
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
