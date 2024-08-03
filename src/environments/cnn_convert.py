import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium import spaces
from highway_env.envs.common.observation import *
from highway_env.vehicle.graphics import VehicleGraphics

from highway_env.envs.common import observation
import sys

class MultiAgentObservationCNN(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe(self) -> tuple:
        #return tuple(obs_type.observe() for obs_type in self.agents_observation_types)
        return tuple(self.observecls(index) for index in range(len(self.agents_observation_types)))

    def observecls(self, index=0) -> np.ndarray:
        # CHANGE COLOUR TO PURPLE
        self.env.controlled_vehicles[index].color = VehicleGraphics.PURPLE
        # OBSERVE
        obs = self.agents_observation_types[index].observe()
        # CHANGE COLOUR BACK
        self.env.controlled_vehicles[index].color = VehicleGraphics.EGO_COLOR
        return obs



def cnn_config_env(env):
    if env.config["observation"]["type"] == "MultiAgentObservation":
        # Monkey patching via sys.modules 
        observation.MultiAgentObservation = MultiAgentObservationCNN

        env.configure(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "GrayscaleObservation",
                        "observation_shape": (128, 64),
                        "stack_size": 4,
                        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                        "scaling": 1.75,
                    }
                },
            }
        )
    else:
        env.configure(
            {
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (128, 64),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                    "scaling": 1.75,
                },
            }
        )
    env.reset()
    return env

def make_cnn_train_env(env_name):
    env = gym.make(env_name)
    cnn_config_env(env)
    return env
