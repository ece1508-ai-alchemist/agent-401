import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from vanilla_continuous_wrapper import ContinuousWrapper

import highway_env  # noqa: F401

TRAIN = False

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
    n_cpu = 6
    batch_size = 64
    
    # Create a vectorized environment with the custom continuous action wrapper
    def make_env():
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        env = ContinuousWrapper(env)
        return env

    env = SubprocVecEnv([make_env for _ in range(n_cpu)])
    
    # Initialize the SAC model
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=batch_size,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log="highway_fast_sac/",
    )
    
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save("highway_fast_sac/model")
        del model  # Remove to demonstrate saving and loading

    # Load the trained model
    model = SAC.load("highway_fast_sac/model", env=env)

    # Create a non-vectorized environment for recording with render_mode="rgb_array"
    env = make_env()
    env = RecordVideo(
        env, video_folder="highway_fast_sac/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    # Evaluate the trained model and record videos
    success_rate = evaluate_agent(env, model, num_episodes=10)

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
    print(f"Success rate: {success_rate * 100:.2f}%")
    env.close()
