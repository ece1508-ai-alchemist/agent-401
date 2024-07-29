import os
import argparse
import gymnasium as gym
from stable_baselines3 import SAC, DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from sb3_utils.callbacks import SaveOnBestTrainingRewardCallback
from sb3_utils.log_collector import convert_events_to_csv
from sb3_utils.misc import load_hyperparameters, cnn_config_env, make_cnn_train_env
import highway_env  # noqa: F401
from environments.register_envs import register_envs


def main(args):
    TRAIN = args.train_steps > 0
    TEST = args.test_runs > 0
    CALLBACK_STEPS = 1000

    # Register custom environments
    register_envs()
    
    # Load hyperparameters
    hyperparameters = load_hyperparameters(args.agent, args.network)
    n_cpu = hyperparameters.pop("n_cpu", 1)

    # Create log dir
    suffix = "_cnn" if args.network == "CNN" else ""
    log_dir = f"out/{args.environment}_{args.agent.lower()}{suffix}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a vectorized environment with render_mode="rgb_array"
    if args.network == "CNN":
        env_cls = lambda: make_cnn_train_env(args.environment)
    else:
        env_cls = lambda: gym.make(args.environment)
    env = make_vec_env(env_cls, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, log_dir)

    # Initialize the model
    if args.agent.upper() == "SAC":
        model_cls = SAC
        policy = "CnnPolicy" if args.network == "CNN" else "MlpPolicy"
    elif args.agent.upper() == "DQN":
        model_cls = DQN
        policy = "CnnPolicy" if args.network == "CNN" else "MlpPolicy"
    elif args.agent.upper() == "PPO":
        model_cls = PPO
        policy = "CnnPolicy" if args.network == "CNN" else "MlpPolicy"
    else:
        raise ValueError(f"Unsupported agent: {args.agent}")

    model = model_cls(
        policy,
        env,
        **hyperparameters,
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=CALLBACK_STEPS, log_dir=log_dir)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=args.train_steps, callback=callback)
        model.save(os.path.join(log_dir, "model"))
        del model  # Remove to demonstrate saving and loading
        convert_events_to_csv(log_dir)

    if TEST:
        # Load the trained model
        model = model_cls.load(os.path.join(log_dir, "model"), env=env)

        # Create a non-vectorized environment for recording with render_mode="rgb_array"
        env = gym.make(args.environment, render_mode="rgb_array")
        if args.network == "CNN":
            cnn_config_env(env)
        env = RecordVideo(
            env, video_folder=log_dir + "videos", episode_trigger=lambda e: True
        )
        env.unwrapped.set_record_video_wrapper(env)

        # Run the trained model and record videos
        for video in range(args.test_runs):
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

if __name__ == "__main__":
    # Random error
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument("environment", type=str, help="The environment name")
    parser.add_argument("agent", type=str, help="The RL agent name")
    parser.add_argument("--train_steps", type=int, default=0, help="Number of training steps")
    parser.add_argument("--test_runs", type=int, default=0, help="Number of test runs")
    parser.add_argument("--hyperparameter", type=str, help="Path to the hyperparameter JSON file")
    parser.add_argument("--network", type=str, choices=["MLP", "CNN"], default="MLP", help="The network type (MLP or CNN)")

    args = parser.parse_args()
    
    if args.hyperparameter:
        hyperparameters = load_hyperparameters(args.hyperparameter, args.network)
    else:
        hyperparameters = load_hyperparameters(args.agent, args.network)
    
    main(args)