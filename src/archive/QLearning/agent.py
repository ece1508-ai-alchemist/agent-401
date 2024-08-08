import glob
import os
import pickle
import random
from base64 import b64encode
from collections import defaultdict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML, display
from tqdm import tqdm


class QLearningAgent:
    def __init__(
        self,
        action_space,
        state_space,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
    ):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def _state_to_key(self, state):
        # Convert state to a hashable type (tuple)
        return tuple(map(tuple, state))

    def choose_action(self, state):
        state_key = self._state_to_key(state)
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        q_predict = self.q_table[state_key][action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (q_target - q_predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Agent saved to {file_name}")

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.q_table = defaultdict(
                lambda: np.zeros(self.action_space.n), pickle.load(f)
            )
        print(f"Agent loaded from {file_name}")


def train_agent(num_episodes=2000, save_file=None):
    # Create the environment
    env = gym.make("highway-fast-v0")
    agent = QLearningAgent(env.action_space, env.observation_space)

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            try:
                agent.learn(state, action, reward, next_state, done)
            except TypeError as e:
                print(f"Episode: {episode}, State: {state}, Next State: {next_state}")
                raise e
            state = next_state
            total_reward += reward

        if (episode + 1) % 100 == 0:
            tqdm.write(
                f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}"
            )

    env.close()

    if save_file:
        agent.save(save_file)

    return agent


def test_agent(agent, num_episodes=10, record=False, video_folder="videos"):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")

    if record:
        env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

    for episode in tqdm(range(num_episodes), desc="Testing Progress"):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

        tqdm.write(
            f"Test Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}"
        )

    env.close()
    if record:
        show_video(video_folder)


def show_video(video_folder):
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    video_files.sort(key=os.path.getmtime)
    if video_files:
        latest_video = video_files[-1]
        video_url = f"{latest_video}"
        video_b64 = b64encode(open(video_url, "rb").read()).decode("ascii")
        display(
            HTML(
                f"""
            <video width="720" height="480" controls>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4" />
            </video>
        """
            )
        )
    else:
        print("No video files found")


def main():
    agent = train_agent(save_file="q_learning_agent.pkl")
    test_agent(agent, record=True)

    # To load and test an existing agent, uncomment the following lines:
    # agent = QLearningAgent(action_space=gym.make('highway-fast-v0')
    # .action_space, state_space=gym.make('highway-fast-v0').observation_space)
    # agent.load("q_learning_agent.pkl")
    # test_agent(agent, record=True)


if __name__ == "__main__":
    main()
