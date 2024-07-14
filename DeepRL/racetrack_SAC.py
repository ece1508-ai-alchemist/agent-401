import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from DeepRL.utils.callbacks import SaveOnBestTrainingRewardCallback
from DeepRL.utils.log_collector import convert_events_to_csv
import highway_env  # noqa: F401

if __name__ == "__main__":
    TRAIN = True
    TEST = False
    n_cpu = 6
    batch_size = 64
    
    # Create log dir 
    log_dir = "racetrack_sac/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a vectorized environment with render_mode="rgb_array"
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={"render_mode": "rgb_array"})
    env = VecMonitor(env, log_dir)

    # Initialize the SAC model
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=batch_size,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log=log_dir,
    )
    
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5), callback=callback)
        model.save(os.path.join(log_dir, "model"))
        del model  # Remove to demonstrate saving and loading
        convert_events_to_csv(log_dir)

    if TEST:
        # Load the trained model
        model = SAC.load(os.path.join(log_dir, "model"), env=env)

        # Create a non-vectorized environment for recording with render_mode="rgb_array"
        env = gym.make("racetrack-v0", render_mode="rgb_array")
        env = RecordVideo(
            env, video_folder="racetrack_sac/videos", episode_trigger=lambda e: True
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
