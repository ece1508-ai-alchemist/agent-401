import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ContinuousWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ContinuousWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.discrete_action_map = {
            0: -1,  # Slow down
            1: 0,   # Maintain speed
            2: 1    # Speed up
        }

    def action(self, action):
        # Map continuous action to discrete actions
        action = action[0]
        if action < -0.33:
            return 0  # Slow down
        elif action < 0.33:
            return 1  # Maintain speed
        else:
            return 2  # Speed up
