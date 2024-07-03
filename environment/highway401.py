from typing import Dict, Text, Tuple

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
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class Highway401(AbstractEnv):
    """
    A Customized Highway environment.

                     |
                     |
    roundabout-----intersection------
      |              |
      |              |
      \             /
        -----------
             /
        -----
    """  # noqa

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [5, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "offroad_terminal": True,
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
        print("crash" + str(self.vehicle.crashed))
        return self.vehicle.crashed or (
            self.config["offroad_terminal"] and not self.vehicle.on_road
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
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        # # Straight line
        # net.add_lane(
        #     "a",
        #     "b",
        #     StraightLane(
        #         [-300, 130],
        #         [-22, 130],
        #         line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        #         speed_limit=10,
        #     ),
        # )
        #
        # net.add_lane(
        #     "a",
        #     "b",
        #     StraightLane(
        #         [-300, 130 + lane_width],
        #         [-170, 130 + lane_width],
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         speed_limit=10,
        #     ),
        # )
        # net.add_lane(
        #     "a",
        #     "b",
        #     StraightLane(
        #         [-170, 130 + lane_width],
        #         [-100, 130 + lane_width],
        #         line_types=(LineType.NONE, LineType.STRIPED),
        #         speed_limit=10,
        #     ),
        # )
        #
        # net.add_lane(
        #     "a",
        #     "b",
        #     StraightLane(
        #         [-100, 130 + lane_width],
        #         [-22, 130 + lane_width],
        #         line_types=(LineType.NONE, LineType.CONTINUOUS),
        #         speed_limit=10,
        #     ),
        # )
        #
        # # merging
        # amplitude = 6
        # ljk = StraightLane([-300, 150], [-250, 150], line_types=(c, c), forbidden=True)
        # lkf = SineLane(
        #     [-250, 144],
        #     [-200, 144],
        #     amplitude,
        #     2 * np.pi / (2 * -50),
        #     np.pi / 2,
        #     line_types=[c, c],
        #     forbidden=True,
        # )
        # net.add_lane(
        #     "f",
        #     "b",
        #     StraightLane(
        #         [-200, 130 + lane_width * 2],
        #         [-100, 130 + lane_width * 2],
        #         line_types=(LineType.NONE, LineType.CONTINUOUS),
        #         speed_limit=10,
        #         forbidden=True,
        #     ),
        # )
        #
        # net.add_lane("j", "k", ljk)
        # net.add_lane("k", "f", lkf)
        #
        # # # 2 - Circular Arc #1
        # center1 = [-22, 110]
        # radii1 = 20
        # net.add_lane(
        #     "b",
        #     "c",
        #     CircularLane(
        #         center1,
        #         radii1,
        #         np.deg2rad(90),
        #         np.deg2rad(-1),
        #         clockwise=False,
        #         line_types=(LineType.CONTINUOUS, LineType.NONE),
        #         speed_limit=10,
        #     ),
        # )
        #
        # net.add_lane(
        #     "b",
        #     "c",
        #     CircularLane(
        #         center1,
        #         radii1 + 4,
        #         np.deg2rad(90),
        #         np.deg2rad(-1),
        #         clockwise=False,
        #         line_types=(LineType.STRIPED, LineType.CONTINUOUS),
        #         speed_limit=10,
        #     ),
        # )
        #
        # for corner in range(4):
        #     angle = np.radians(90 * corner)
        #     is_horizontal = corner % 2
        #     priority = 3 if is_horizontal else 1
        #     rotation = np.array(
        #         [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        #     )
        #     # Incoming
        #     start = rotation @ np.array(
        #         [lane_width / 2, access_length + outer_distance]
        #     )
        #     end = rotation @ np.array([lane_width / 2, outer_distance])
        #     net.add_lane(
        #         "o" + str(corner),
        #         "ir" + str(corner),
        #         StraightLane(
        #             start,
        #             end,
        #             line_types=[s, c],
        #             priority=priority,
        #             speed_limit=10,
        #         ),
        #     )
        #     # Right turn
        #     r_center = rotation @ (np.array([outer_distance, outer_distance]))
        #     net.add_lane(
        #         "ir" + str(corner),
        #         "il" + str((corner - 1) % 4),
        #         CircularLane(
        #             r_center,
        #             right_turn_radius,
        #             angle + np.radians(180),
        #             angle + np.radians(270),
        #             line_types=[n, c],
        #             priority=priority,
        #             speed_limit=10,
        #         ),
        #     )
        #     # Left turn
        #     l_center = rotation @ (
        #         np.array(
        #             [
        #                 -left_turn_radius + lane_width / 2,
        #                 left_turn_radius - lane_width / 2,
        #             ]
        #         )
        #     )
        #     net.add_lane(
        #         "ir" + str(corner),
        #         "il" + str((corner + 1) % 4),
        #         CircularLane(
        #             l_center,
        #             left_turn_radius,
        #             angle + np.radians(0),
        #             angle + np.radians(-90),
        #             clockwise=False,
        #             line_types=[n, n],
        #             priority=priority - 1,
        #             speed_limit=10,
        #         ),
        #     )
        #     # Straight
        #     start = rotation @ np.array([lane_width / 2, outer_distance])
        #     end = rotation @ np.array([lane_width / 2, -outer_distance])
        #     net.add_lane(
        #         "ir" + str(corner),
        #         "il" + str((corner + 2) % 4),
        #         StraightLane(
        #             start, end, line_types=[s, n], priority=priority, speed_limit=10
        #         ),
        #     )
        #     # Exit
        #     start = rotation @ np.flip(
        #         [lane_width / 2, access_length + outer_distance], axis=0
        #     )
        #     end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
        #     net.add_lane(
        #         "il" + str((corner - 1) % 4),
        #         "o" + str((corner - 1) % 4),
        #         StraightLane(
        #             end, start, line_types=[n, c], priority=priority, speed_limit=10
        #         ),
        #     )
        #
        # dev = 20 # [m]
        # a = 5  # [m]
        # delta_st = 0.2 * dev  # [m]
        # w = 2 * np.pi / 5
        # delta_en = dev - delta_st
        # # enter to roundabout
        # net.add_lane(
        #     "il1",
        #     "ee",
        #     SineLane(
        #         [-111, -6],
        #         [-121, -6],
        #         4,
        #         w,
        #         -np.pi / 2,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "ex",
        #     "o1",
        #     SineLane(
        #         [-120, 6],
        #         [-111, 6],
        #         4,
        #         w,
        #         18.6,
        #         line_types=(c, c),
        #     ),
        # )
        # self._add_roundabout(net)

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        lane_h_end, lane_v_end = self._add_straight_highway_with_merging(road)
        self._add_road_to_enter_intersection(road, lane_h_end)

        self.road = road

    @staticmethod
    def _add_straight_highway_with_merging(road: RegulatedRoad) -> Tuple[int, int]:
        net = road.network
        lane_star = 0
        merging_line_vertical_distance = 15
        lane_width = AbstractLane.DEFAULT_WIDTH
        line_length = 150
        amplitude = 4
        merging_straight_line_length = line_length - 50
        merging_sine_line_length = 39
        merging_straight_line2_length = merging_straight_line_length

        # ab1 straight line 0
        ab_0 = StraightLane(
            [lane_star, lane_star],
            [line_length, lane_star],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        net.add_lane("a", "b", ab_0)

        ab_1 = StraightLane(
            [lane_star, lane_star + lane_width],
            [line_length, lane_star + lane_width],
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
            [lane_star, merging_line_vertical_distance],
            [merging_straight_line_length, merging_line_vertical_distance],
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
            [merging_straight_line_length, merging_line_vertical_distance - amplitude],
            [m23_end, merging_line_vertical_distance - amplitude],
            amplitude,
            2 * np.pi / (2 * -50),
            np.pi / 2,
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
            forbidden=True,
        )
        net.add_lane("m2", "b", m23)

        # bc straight merging line 2
        merging_line_end = m23_end + merging_straight_line2_length
        m34 = StraightLane(
            [m23_end, lane_star + 2 * lane_width],
            [merging_line_end, lane_star + 2 * lane_width],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            forbidden=True,
        )
        net.add_lane("b", "c", m34)

        # bc straight line 0
        bc_0 = StraightLane(
            [line_length, lane_star],
            [merging_line_end, lane_star],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        net.add_lane("b", "c", bc_0)

        bc_1 = StraightLane(
            [lane_star, lane_star + lane_width],
            [merging_line_end, lane_star + lane_width],
            line_types=(LineType.NONE, LineType.STRIPED),
        )
        # bc straight line 1
        net.add_lane(
            "b",
            "c",
            bc_1,
        )

        # cd straight line 0
        cd_0 = StraightLane(
            [merging_line_end, lane_star],
            [merging_line_end + line_length, lane_star],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            speed_limit=10,
        )
        cd_1 = StraightLane(
            [merging_line_end, lane_star + lane_width],
            [merging_line_end + line_length, lane_star + lane_width],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=10,
        )
        net.add_lane("c", "d", cd_0)
        net.add_lane("c", "d", cd_1)

        road.network = net
        road.objects.append(
            Obstacle(
                road,
                [m23_end + merging_straight_line2_length, lane_star + 2 * lane_width],
            )
        )
        return merging_line_end + line_length, int(lane_star + lane_width)

    @staticmethod
    def _add_road_to_enter_intersection(road: RegulatedRoad, h_start: int):
        net = road.network
        lane_length = 20
        lane_width = AbstractLane.DEFAULT_WIDTH
        radii1 = 20
        center1 = [h_start, -radii1]
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                clockwise=False,
                line_types=[LineType.CONTINUOUS, LineType.NONE],
                speed_limit=10,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center1,
                radii1 + lane_width,
                np.deg2rad(90),
                np.deg2rad(-1),
                clockwise=False,
                line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                speed_limit=10,
            ),
        )

        h_end = h_start + radii1
        v_end = -radii1 - lane_length

        net.add_lane(
            "e",
            "f",
            StraightLane(
                [h_end, -radii1],
                [h_end, v_end],
                line_types=[LineType.CONTINUOUS, LineType.STRIPED],
                speed_limit=10,
            ),
        )
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [h_end + lane_width, -radii1],
                [h_end + lane_width, v_end],
                line_types=[LineType.NONE, LineType.CONTINUOUS],
                speed_limit=10,
            ),
        )
        road.network = net
        return h_end, v_end

    def _add_intersection_to_road(
        self, road: RegulatedRoad, h_start: int, v_start: int
    ): ...

    def _add_roundabout(self, net: RoadNetwork):
        center = [-145, 0]  # [m]
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

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=10
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
        #     lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
        #     position = lane.position(position + self.np_random.uniform(-5, 5), 0)
        #     speed += self.np_random.uniform(-1, 1)
        #     road.vehicles.append(other_vehicles_type(road, position, speed=speed))
        #
        # merging_v = other_vehicles_type(
        #     road, road.network.get_lane(("m1", "m2", 0)).position(110, 0), speed=20
        # )
        # merging_v.target_speed = 30
        # road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
