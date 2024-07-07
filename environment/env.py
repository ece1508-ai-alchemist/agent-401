import gymnasium as gym
from gymnasium.envs.registration import register
from matplotlib import pyplot as plt

register(id="401-v0", entry_point="environment.highway401:Highway401")
action_type = "DiscreteMetaAction"  # ContinuousAction or DiscreteMetaAction
env = gym.make("401-v0", render_mode="rgb_array")

env.configure({"action": {"type": action_type}})
env.reset()
done = False


while not done:
    if action_type == "ContinuousAction":
        action = env.action_space.sample()
    else:
        action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()
