import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append(os.getcwd())
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from DeepRL.utils.callbacks import SaveOnBestTrainingRewardCallback
from DeepRL.utils.log_collector import convert_events_to_csv
import highway_env  # noqa: F401

# Import your custom environment
from environment.highway401 import Highway401

# Register the custom environment
register(id="401-v0", entry_point="environment.highway401:Highway401")

def highway401_train_env():
    env = gym.make("401-v0")
    return env

if __name__ == "__main__":
    TRAIN = True
    TEST = True
    n_cpu = 6
    batch_size = 64
    
    # Create log dir 
    log_dir = "highway401_ppo/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a vectorized environment with render_mode="rgb_array"
    env = make_vec_env(highway401_train_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, log_dir)

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=2,
        tensorboard_log=log_dir,
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
        model = PPO.load(os.path.join(log_dir, "model"), env=env)

        # Create a non-vectorized environment for recording with render_mode="rgb_array"
        env = gym.make("401-v0", render_mode="rgb_array")
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
        env.close()
