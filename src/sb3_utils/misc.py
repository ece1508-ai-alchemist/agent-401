import os
import json

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
