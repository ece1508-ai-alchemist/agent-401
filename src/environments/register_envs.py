import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env  # noqa: F401
from environments.highway401 import Highway401

def register_envs():
    # Put all the register calls here
    register(id="401-v0", entry_point="environments.highway401:Highway401")