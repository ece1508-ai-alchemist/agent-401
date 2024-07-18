import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import CnnPolicy
import torch
import highway_env  # noqa: F401

def make_env():
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.587, 0.114],  # Weights for converting RGB to grayscale
            "stack_size": 4,  # Number of frames to stack
            "observation_shape": (84, 84)  # Shape of the resulting image
        },
        "action": {
            "type": "ContinuousAction",
        },
        # Add other environment configurations if needed
    }
    env = gym.make("racetrack-v0", render_mode="rgb_array", config=config)
    return env

def evaluate_agent(env, model, num_episodes=10):
    successes = 0
    for _ in range(num_episodes):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        if not info.get('crashed', False):
            successes += 1
    success_rate = successes / num_episodes
    print(f"Success rate: {success_rate * 100:.2f}%")
    return success_rate

if __name__ == "__main__":
    TRAIN = True
    batch_size = 64
    
    # Create the environment
    env = make_env()

    # Initialize the SAC model with a CNN policy
    model = SAC(
        CnnPolicy,
        env,
        batch_size=batch_size,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log="racetrack_sac/",
        device="cuda"  # Set the device to CUDA
    )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save("racetrack_sac/model")
        del model  # Remove to demonstrate saving and loading

    # Load the trained model
    model = SAC.load("racetrack_sac/model", env=env)

    # Create a non-vectorized environment for recording with render_mode="rgb_array"
    env = make_env()
    env = RecordVideo(
        env, video_folder="racetrack_sac/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    # Evaluate the trained model and record videos
    success_rate = evaluate_agent(env, model, num_episodes=10)
    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict the action
            action, _states = model.predict(obs, deterministic=True)
            # Perform the action and get the next state
            obs, reward, done, truncated, info = env.step(action)
            # Render the environment
            env.render()
    env.close()