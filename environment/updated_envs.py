from typing import Dict, Text, Optional

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
    
    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        # Storage Val
        self.prev_lane_idx = None
        self.prev_best_distance = None # Some very high value
        self.prev_position = None
        
        super().__init__(config, render_mode)

        

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "target_speeds": [0, 8, 16]
            },
            #"offroad_terminal": True,
            "incoming_vehicle_destination": None,
            "collision_reward": -1, # Never crash
            "high_speed_reward": 0.2,
            "lane_centering_cost": 4,
            "lane_centering_reward": 1, # Reward for staying in center
            "lane_change_reward": -0.05, # Penalty for changing lanes unnecessarily
            "action_reward": -0.3, # Penalty for doing unnecessary actions
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "progress_reward": 0.3, # Try to move closer
            "distance_reward": 1, # Closer is better
            "backtrack_reward": -0.3, # Do not backtrack 
            "normalize_reward": True #TODO - Do we need this?
        })
        return config
    

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1]) #TODO - set better interval?
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:

        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if self.prev_lane_idx is not None and self.prev_lane_idx != self.vehicle.lane_index:
            lane_change_reward = 1
        else:
            lane_change_reward = 0

        self.prev_lane_idx = self.vehicle.lane_index

        distance = np.linalg.norm(self.vehicle.destination-self.vehicle.position)
        if self.prev_best_distance is not None and distance < self.prev_best_distance:
            progress_reward = 1 # Reward for making progress
        else:
            progress_reward = 0

        if self.prev_best_distance is not None and distance > self.prev_best_distance:
            backtrack_reward = 1
        else:
            backtrack_reward = 0

        if self.prev_best_distance is None:
            self.prev_best_distance = distance
        else:
            self.prev_best_distance = min(self.prev_best_distance, distance)
        #print("destination", self.vehicle.destination, "position", self.vehicle.position, "direction", self.vehicle.destination_direction, "distance", np.linalg.norm(self.vehicle.destination-self.vehicle.position))
        return {
            "lane_centering_reward": 1/(1+self.config["lane_centering_cost"]*lateral**2),
            "high_speed_reward":
                 MDPVehicle.get_speed_index(self.vehicle) / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1), 
            #"action_reward": np.linalg.norm(action),
            "lane_change_reward": lane_change_reward,
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
            "distance_reward": 1 / (distance + 1),
            "progress_reward": progress_reward, 
            "backtrack_reward": backtrack_reward
        }
    
    def _is_terminated(self) -> bool:
        return (
            self.vehicle.crashed or
            not self.vehicle.on_road
        )
    
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(125, 0),
                                                     speed=8,
                                                     heading=ego_lane.heading_at(140))
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            try:
                path = ego_vehicle.road.network.shortest_path(ego_vehicle.lane_index[1], "nxs")
            except KeyError:
                path = []
            if path:
                #print("success", path)
                ego_vehicle.route = [ego_vehicle.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
            else:
                ego_vehicle.route = [ego_vehicle.lane_index]
            #print("success", ego_vehicle.route)
        
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("we", "sx", 1),
                                                   longitudinal=5 + self.np_random.normal()*position_deviation,
                                                   speed=16 + self.np_random.normal() * speed_deviation)

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("we", "sx", 0),
                                                       longitudinal=20*i + self.np_random.normal()*position_deviation,
                                                       speed=16 + self.np_random.normal() * speed_deviation)
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("eer", "ees", 0),
                                                   longitudinal=50 + self.np_random.normal() * position_deviation,
                                                   speed=16 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

