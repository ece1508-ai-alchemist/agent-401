import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

TRAIN = True

if __name__ == "__main__":
    n_cpu = 6
    batch_size = 64
    
    # Create a vectorized environment with render_mode="rgb_array"
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={"render_mode": "rgb_array"})
    
    # Initialize the SAC model
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=batch_size,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log="racetrack_sac/",
    )
    
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save("racetrack_sac/model")
        del model  # Remove to demonstrate saving and loading

    # Load the trained model
    model = SAC.load("racetrack_sac/model", env=env)

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
