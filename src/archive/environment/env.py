import gymnasium as gym
import os
import sys
sys.path.append(os.getcwd())
from gymnasium.envs.registration import register
# from environment.updated_envs import ModifiedMergeEnv, ModifiedRoundaboutEnv  # noqa
from matplotlib import pyplot as plt

register(id="401-v0", entry_point="environment.highway401:Highway401")

highway401_env = gym.make("401-v0", render_mode="rgb_array")

selected_env = highway401_env
selected_env.configure({
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (84, 84),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],

    },
    "controlled_vehicles": 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "lateral": False,
            "longitudinal": True,
        },
    },
    "offroad_terminal": True,
})

selected_env.reset()
done = False

while not done:
    action = selected_env.action_space.sample()

    obs, reward, done, truncated, info = selected_env.step(action)
    selected_env.render()

plt.imshow(selected_env .render())
plt.show()
