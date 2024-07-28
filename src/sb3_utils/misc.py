import os
import json
import gymnasium as gym
import highway_env  # noqa: F401

def load_hyperparameters(agent_name, network_type):
    suffix = "_cnn" if network_type == "CNN" else ""
    # Default file path
    hyperparam_path = f"sb3_model_hyperparameters/{agent_name.lower()}{suffix}.json"
    
    if os.path.isfile(hyperparam_path):
        with open(hyperparam_path, 'r') as file:
            hyperparameters = json.load(file)
    else:
        raise FileNotFoundError(f"Hyperparameter file {hyperparam_path} not found.")
    
    return hyperparameters

def cnn_config_env(env):
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
