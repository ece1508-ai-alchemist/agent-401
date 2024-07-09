import gymnasium as gym
from gymnasium.envs.registration import register
from environment.updated_envs import ModifiedMergeEnv, ModifiedRoundaboutEnv  # noqa
from matplotlib import pyplot as plt


register(id="401-v0", entry_point="environment.highway401:Highway401")
register(id="merge-v1", entry_point="environment.updated_envs:ModifiedMergeEnv")
register(id="roundabout-v1", entry_point="environment.updated_envs:ModifiedRoundaboutEnv")

action_type = "ContinuousAction"  # ContinuousAction or DiscreteMetaAction

roundabout_env = gym.make("roundabout-v1", render_mode="rgb_array")
merge_env = gym.make("merge-v1", render_mode="rgb_array")
highway401_env = gym.make("401-v0", render_mode="rgb_array")

selected_env = merge_env
selected_env .configure({
    "action": {"type": action_type},
    "offroad_terminal": True,
})
selected_env .reset()
done = False


while not done:
    if action_type == "ContinuousAction":
        action = selected_env .action_space.sample()
    else:
        action = selected_env .action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = selected_env .step(action)
    selected_env .render()

plt.imshow(selected_env .render())
plt.show()
