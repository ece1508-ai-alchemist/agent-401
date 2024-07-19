import os
import sys
sys.path.append(os.getcwd())
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from DeepRL.utils.callbacks import SaveOnBestTrainingRewardCallback
from DeepRL.utils.log_collector import convert_events_to_csv
import highway_env  # noqa: F401

def intersection_mlp_config_env(env):
    env.configure(
        {
            "observation": {
                "type": "Kinematics",  # Use the original observation type
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"]
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            # Add other environment configurations if needed
        }
    )
    env.reset()
    return env

def intersection_mlp_train_env():
    env = gym.make("intersection-v0")
    intersection_mlp_config_env(env)
    return env

if __name__ == "__main__":
    TRAIN = True
    TEST = True
    n_cpu = 6
    batch_size = 64
    
    # Create log dir 
    log_dir = "intersection_mlp_dqn/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a vectorized environment with the original observation space
    env = make_vec_env(intersection_mlp_train_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, log_dir)

    # Initialize the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda"  # Set the device to CUDA
    )
    
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4), callback=callback)
        model.save(os.path.join(log_dir, "model"))
        del model  # Remove to demonstrate saving and loading
        convert_events_to_csv(log_dir)

    if TEST:
        # Load the trained model
        model = DQN.load(os.path.join(log_dir, "model"), env=env)

        # Create a non-vectorized environment for recording with render_mode="rgb_array"
        env = gym.make("intersection-v0", render_mode="rgb_array")
        intersection_mlp_config_env(env)
        env = RecordVideo(
            env, video_folder=log_dir + "videos", episode_trigger=lambda e: True
        )
        env.unwrapped.set_record_video_wrapper(env)

        # Run the trained model and record videos
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
            env.close_video_recorder()
        env.close()
