import gymnasium as gym
import highway_env  # noqa: F401

def cnn_config_env(env):
    if env.config["observation"]["type"] == "MultiAgentObservation":
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
