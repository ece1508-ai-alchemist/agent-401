from gymnasium.envs.registration import register
from environment.updated_envs import ModifiedMergeEnv, ModifiedRoundaboutEnv  # noqa
register(
    id="ModifiedRoundaboutEnv-v0",
    entry_point="environment.updated_envs:ModifiedRoundaboutEnv",
)