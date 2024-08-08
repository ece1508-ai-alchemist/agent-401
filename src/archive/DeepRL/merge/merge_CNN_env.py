
import gymnasium as gym
import highway_env  # noqa: F401

def merge_cnn_config_env(env):
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


def merge_cnn_train_env():
    env = gym.make("merge-v0")
    merge_cnn_config_env(env)
    return env


