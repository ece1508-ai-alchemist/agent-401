import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from DeepRL.utils.callbacks import SaveOnBestTrainingRewardCallback
import highway_env  # noqa: F401
from environment.updated_envs import ModifiedMergeEnv, ModifiedRoundaboutEnv  # noqa


if __name__ == "__main__":
    TRAIN = True
    TEST = True
    n_cpu = 6
    batch_size = 64
    
    # Create log dir
    log_dir = "roundabout_SAC/"
    os.makedirs(log_dir, exist_ok=True)

    # Default environments
    #register_highway_envs()
  
    #roundabout_env = gym.make("ModifiedRoundaboutEnv-v0", render_mode="rgb_array")
    
    # Create a vectorized environment with render_mode="rgb_array"
    env = make_vec_env("ModifiedRoundaboutEnv-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={"render_mode": "rgb_array"})
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
        model.learn(total_timesteps=int(3e5), callback=callback)
        model.save(os.path.join(log_dir, "model"))
        del model  # Remove to demonstrate saving and loading

    if TEST:
        # Load the trained model
        model = SAC.load(os.path.join(log_dir, "model"), env=env)

        # Create a non-vectorized environment for recording with render_mode="rgb_array"
        env = gym.make("ModifiedRoundaboutEnv-v0", render_mode="rgb_array")
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