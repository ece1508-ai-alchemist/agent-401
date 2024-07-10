from typing import Dict, Text

import numpy as np
from highway_env import utils
from highway_env import register_highway_envs
from highway_env.envs.merge_env import MergeEnv
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle


class ModifiedMergeEnv(MergeEnv):
    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        if isinstance(action, (int, np.int_)):

            lane_change_reward = action in [0, 2]
        else:
            lane_change_reward = action[1] != 0

        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": lane_change_reward,
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }


class ModifiedRoundaboutEnv(RoundaboutEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": True,
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "ContinuousAction",
                "target_speeds": [0, 8, 16]
            },
            "offroad_terminal": True,
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True
        })
        return config
    
    def _rewards(self, action: int) -> Dict[Text, float]:
        if isinstance(action, (int, np.int_)):

            lane_change_reward = 0
        else:
            lane_change_reward = 0
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
                                 / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward":  lane_change_reward,
            "on_road_reward": self.vehicle.on_road,
        }
