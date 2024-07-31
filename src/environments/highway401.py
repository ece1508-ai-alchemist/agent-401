from typing import Dict, List, Text, Tuple

import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import (
    AbstractLane,
    CircularLane,
    LineType,
    SineLane,
    StraightLane,
)
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetwork401(RoadNetwork):
    def position_heading_along_route(
            self,
            route: Route,
            longitudinal: float,
            lateral: float,
            current_lane_index: LaneIndex = None,  # Ensure it has a default value
    ) -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: lateral position
        :param current_lane_index: current lane index of the vehicle
        :return: position, heading
        """
        
        def _get_route_head_with_id(route_):
            lane_index_ = route_[0]
            if lane_index_[2] is None:
                # Handle cases where current_lane_index is None
                if current_lane_index is None:
                    id_ = 0
                else:
                    if len(self.graph[lane_index_[0]][lane_index_[1]]) == 1:
                        id_ = 0
                    elif len(self.graph[lane_index_[0]][lane_index_[1]]) < len(
                            self.graph[current_lane_index[0]][current_lane_index[1]]
                    ):
                        id_ = len(self.graph[lane_index_[0]][lane_index_[1]]) - 1
                    else:
                        id_ = (
                            current_lane_index[2]
                            if current_lane_index and current_lane_index[2] < len(self.graph[current_lane_index[0]][current_lane_index[1]])
                            else 0
                        )
                lane_index_ = (lane_index_[0], lane_index_[1], id_)
            return lane_index_

        lane_index = _get_route_head_with_id(route)
        while len(route) > 1 and longitudinal > self.get_lane(lane_index).length:
            longitudinal -= self.get_lane(lane_index).length
            route = route[1:]
            lane_index = _get_route_head_with_id(route)

        return self.get_lane(lane_index).position(longitudinal, lateral), self.get_lane(
            lane_index
        ).heading_at(longitudinal)


class Highway401(AbstractEnv):
    """
    A Customized Highway environment.

                     |
                     |
    roundabout-----intersection------
      |              |
      |              |
                    /
        -----------
             /
        -----
    """  # noqa

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {"type": "DiscreteMetaAction"},
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "controlled_vehicles": 1,
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "screen_width": 800,
                "screen_height": 800,
                "offroad_terminal": True,
                "destination": "sxr",
                "other_vehicles_destinations": [
                    "o1",
                    "o2",
                    "sxs",
                    "sxr",
                    "exs",
                    "exr",
                    "nxs",
                    "nxr",
                ],
                "spawn_probability": 0.5,
                "duration": 60,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
                any(vehicle.crashed for vehicle in self.controlled_vehicles)
                or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
                or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork401()

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        self._add_intersection_to_road(road, 0, 0)

        lane_h_end, lane_v_end = self._add_straight_highway_with_merging(road)

        lane_h_end, lane_v_end = self._add_road_to_enter_intersection(
            road, lane_h_end, lane_v_end
        )

        self._add_roundabout(road, lane_h_end, lane_v_end)

        self.road = road

    @staticmethod
    def _add_straight_highway_with_merging(road: RegulatedRoad) -> Tuple[int, int]:
        net = road.network
        lane_star = [-411, 129]
        merging_line_vertical_distance = 15
        lane_width = AbstractLane.DEFAULT_WIDTH
        line_length = 150
        amplitude = 4
        merging_straight_line_length = line_length - 50
        merging_sine_line_length = 39
        merging_straight_line2_length = merging_straight_line_length

        # ab1 straight line 0
        ab_0 = StraightLane(
            [lane_star[0], lane_star[1]],
            [lane_star[0] + line_length, lane_star[1]],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        net.add_lane("a", "b", ab_0)

        ab_1 = StraightLane(
            [lane_star[0], lane_star[1] + lane_width],
            [lane_star[0] + line_length, lane_star[1] + lane_width],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
        )
        # ab1 straight line 1
        net.add_lane(
            "a",
            "b",
            ab_1,
        )

        # m1 straight merging line
        m12 = StraightLane(
            [lane_star[0], lane_star[1] + merging_line_vertical_distance],
            [
                lane_star[0] + merging_straight_line_length,
                lane_star[1] + merging_line_vertical_distance,
            ],
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            forbidden=True,
        )
        net.add_lane(
            "m1",
            "m2",
            m12,
        )

        # m2 sin merging line to b
        m23_end = merging_straight_line_length + merging_sine_line_length
        m23 = SineLane(
            [
                lane_star[0] + merging_straight_line_length,
                merging_line_vertical_distance - amplitude + lane_star[1],
            ],
            [
                lane_star[0] + m23_end,
                merging_line_vertical_distance - amplitude + lane_star[1],
            ],
            amplitude,
            2 * np.pi / (2 * -50),
            np.pi / 2,
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
            forbidden=True,
            priority=3,
        )
        net.add_lane("m2", "b", m23)

        # bc straight merging line 2
        merging_line_end = m23_end + merging_straight_line2_length

        # bc straight line 0
        bc_0 = StraightLane(
            [lane_star[0] + line_length, lane_star[1]],
            [lane_star[0] + merging_line_end, lane_star[1]],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        net.add_lane("b", "c", bc_0)

        bc_1 = StraightLane(
            [lane_star[0], lane_star[1] + lane_width],
            [lane_star[0] + merging_line_end, lane_star[1] + lane_width],
            line_types=(LineType.NONE, LineType.STRIPED),
        )
        # bc straight line 1
        net.add_lane(
            "b",
            "c",
            bc_1,
        )

        m34 = StraightLane(
            [lane_star[0] + m23_end, 2 * lane_width + lane_star[1]],
            [lane_star[0] + merging_line_end, 2 * lane_width + lane_star[1]],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            forbidden=True,
            priority=1,
        )
        net.add_lane("b", "c", m34)

        # cd straight line 0
        cd_0 = StraightLane(
            [lane_star[0] + merging_line_end, lane_star[1]],
            [lane_star[0] + merging_line_end + line_length, lane_star[1]],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        cd_1 = StraightLane(
            [lane_star[0] + merging_line_end, lane_star[1] + lane_width],
            [lane_star[0] + merging_line_end + line_length, lane_star[1] + lane_width],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
        )
        net.add_lane("c", "d", cd_0)
        net.add_lane("c", "d", cd_1)

        road.network = net
        road.objects.append(
            Obstacle(
                road,
                [
                    lane_star[0] + m23_end + merging_straight_line2_length,
                    lane_star[1] + 2 * lane_width,
                ],
            )
        )
        return lane_star[0] + merging_line_end + line_length, int(
            lane_star[1] + lane_width
        )

    @staticmethod
    def _add_road_to_enter_intersection(
            road: RegulatedRoad, h_start: int, v_start: int
    ):
        net = road.network
        lane_width = AbstractLane.DEFAULT_WIDTH
        radii1 = 20
        center1 = [h_start, v_start - radii1 - lane_width]
        net.add_lane(
            "d",
            "o0",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                clockwise=False,
                line_types=[LineType.CONTINUOUS, LineType.NONE],
                speed_limit=25,
                priority=3,
            ),
        )
        net.add_lane(
            "d",
            "o0",
            CircularLane(
                center1,
                radii1 + lane_width,
                np.deg2rad(90),
                np.deg2rad(-1),
                clockwise=False,
                line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                speed_limit=25,
                priority=3,
            ),
        )

        h_end = h_start + radii1
        v_end = v_start - radii1 - 11

        road.network = net
        return h_end, v_end

    def _add_intersection_to_road(
            self, road: RegulatedRoad, h_start: int, v_start: int
    ) -> Tuple[int, int]:
        """
               K
               |
        h------j--------i
              |
              g
              |
              f
        """
        net = road.network
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=15
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10,
                ),
            )
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=15
                ),
            )
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=15
                ),
            )

        road.network = net
        return 0, 0

    def _add_roundabout(
            self, road: RegulatedRoad, h_start: int, v_start: int
    ) -> Tuple[int, int]:
        net = road.network
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [-250, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                "se",
                "ex",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(alpha),
                    np.deg2rad(-alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 + alpha),
                    np.deg2rad(-90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ne",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 - alpha),
                    np.deg2rad(-180 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-180 + alpha),
                    np.deg2rad(-180 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "we",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(180 - alpha),
                    np.deg2rad(90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 + alpha),
                    np.deg2rad(90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # Access lanes: (r)oad/(s)ine
        access = 150  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev

        # O
        # |
        # v
        net.add_lane(
            "ser",
            "ses",
            StraightLane(
                [center[0] + 2, access - 50],
                [center[0] + 2, dev / 2],
                line_types=(s, c),
            ),
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [center[0] + 2 + a, dev / 2],
                [center[0] + 2 + a, dev / 2 - delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [center[0] - 2 - a, -dev / 2 + delta_en],
                [center[0] - 2 - a, dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs",
            "sxr",
            StraightLane(
                [center[0] - 2, dev / 2],
                [center[0] - 2, access - 50],
                line_types=(n, c),
            ),
        )

        # O <---
        net.add_lane(
            "o1",
            "ees",
            StraightLane(
                [center[0] + access, -2], [center[0] + dev / 2, -2], line_types=(s, c)
            ),
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [center[0] + dev / 2, -2 - a],
                [center[0] + dev / 2 - delta_st, -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [center[0] + -dev / 2 + delta_en, 2 + a],
                [center[0] + dev / 2, 2 + a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs",
            "exr",
            StraightLane(
                [center[0] + dev / 2, 2], [center[0] + access, 2], line_types=(n, c)
            ),
        )

        net.add_lane(
            "ner",
            "nes",
            StraightLane(
                [center[0] - 2, -access], [center[0] - 2, -dev / 2], line_types=(s, c)
            ),
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [center[0] - 2 - a, -dev / 2],
                [center[0] - 2 - a, -dev / 2 + delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [center[0] + 2 + a, dev / 2 - delta_en],
                [center[0] + 2 + a, -dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs",
            "nxr",
            StraightLane(
                [center[0] + 2, -dev / 2], [center[0] + 2, -access], line_types=(n, c)
            ),
        )

        road.network = net
        return 0, 0

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # 
        for position, speed in [(5, 20), (50, 25), (150, 15), (0, 30)]:
            if self.np_random.uniform() < self.config["spawn_probability"]:
                lane = road.network.get_lane(("a", "b", np.random.choice(range(2))))
                position = lane.position(position + self.np_random.uniform(-2, 2), 0)
                speed += self.np_random.uniform(-1, 1)
                v = other_vehicles_type(road, position, speed=speed)
                v.plan_route_to(
                    np.random.choice(self.config["other_vehicles_destinations"])
                )
                road.vehicles.append(v)

        destination = self.config["destination"] or "o" + str(
            self.np_random.integers(1, 4)
        )

        start_lanes = [("m1", "m2", 0), ("a", "b", 0), ("a", "b", 1)]

        self.controlled_vehicles = []

        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                start_lanes[ego_id]
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(30, 0),
                speed=10,
            )
            try:
                ego_vehicle.plan_route_to(destination)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)

    def _spawn_vehicle(
            self,
            longitudinal: float = 0,
            position_deviation: float = 1.0,
            speed_deviation: float = 1.0,
            spawn_probability: float = 0.6,
            go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o3", "ir3", 0),
            longitudinal=(
                    longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to(
            np.random.choice(self.config["other_vehicles_destinations"])
        )
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
                            and "o" in vehicle.lane_index[1]
                            and vehicle.lane.local_coordinates(vehicle.position)[0]
                            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
               or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
                "sxr" in vehicle.lane_index[1]
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )
