# Q-learning - Off-policy TD control.
import numpy as np
from itertools import count
from value_based_methods.utils import tabular_epsilon_greedy
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym

# def q_learning_update(q_table, state, action, reward, next_state, terminated, gamma, lr):
#     current_value = q_table[state][action]
#     if terminated:
#         target = reward
#     else:
#         optimal_value = np.max(q_table[next_state])
#         target = reward + gamma*optimal_value
#     q_table[state][action] += lr*(target-current_value)
#     return q_table


# def q_learning_training(env, num_episodes, gamma=0.99, epsilon_min=0.05, epsilon_max=1, epsilon_decay=5e-5, lr_min=0.05, lr_max=1, lr_decay=5e-5):
#     q_table = np.zeros((env.observation_space.n, env.action_space.n))
#     episodes = []
#     durations = []
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         epsilon = max(epsilon_min, epsilon_max*np.exp(-epsilon_decay*episode))
#         lr = max(lr_min, lr_max*np.exp(-lr_decay*episode))
#         for i in count():
#             action = tabular_epsilon_greedy(q_table, state, epsilon, env.action_space.n)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             q_table = q_learning_update(q_table, state, action, reward, next_state, terminated, gamma, lr)
#             state = next_state
#             done = terminated or truncated
#             if done:
#                 if episode % 10000 == 0:
#                     print(f'Epsiode: {episode}')
#                 episodes.append(episode)
#                 durations.append(i)
#                 break
#     env.close()
#     return q_table, episodes, durations


class Q_Learning():
    def __init__(self, env_id, episodes_train, episodes_eval, gamma, epsilon_max, epsilon_min, epsilon_decay, lr_max, lr_min, lr_decay, **env_kwargs):
        self.env_kwargs = env_kwargs
        # Environment
        self.env_id = env_id
        self.env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        self.observation_space = self.env.observation_space
        self.action_space =  self.env.action_space 
        # Epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        # Learning Rate
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_decay = lr_decay
        # Stats
        self.episodes = []
        self.durations = []

        self.gamma = gamma
        self.episodes_train = episodes_train
        self.episodes_eval = episodes_eval
        try:
            self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        except:
            raise ValueError("Environment must have discrete observation and action spaces.")

    def train(self):
        for episode in tqdm(range(self.episodes_train)):
            epsilon = max(self.epsilon_min, self.epsilon_max*np.exp(-self.epsilon_decay*episode))
            lr = max(self.lr_min, self.lr_max*np.exp(-self.lr_decay*episode))
            state, _ = self.env.reset()
            for i in count():
                action = tabular_epsilon_greedy(self.q_table, state, epsilon, self.action_space.n)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update_value(state, action, reward, next_state, terminated, lr)
                done = terminated or truncated
                if done:
                    self.episodes.append(episode)
                    self.durations.append(i)
                    break
        self.env.close()

    def update_value(self, state, action, reward, next_state, terminated, lr):
        optimal_future_value = np.max(self.q_table[next_state])
        current_value = self.q_table[state][action]
        target = reward + (not terminated)*self.gamma*optimal_future_value
        self.q_table[state][action] = current_value + lr*(target-current_value)

    def eval(self):
        env = gym.make(self.env_id, render_mode="human", **self.env_kwargs)
        for i in range(self.episodes_eval):
            state, _ = env.reset()
            while True:
                action = np.argmax(self.q_table[state])
                next_state, _, terminated, truncated, _ = env.step(action)
                state = next_state
                done = terminated or truncated
                if done:
                    break

    def plot(self):
        plt.plot(self.episodes, self.durations)
        plt.xlabel("Episodes")
        plt.ylabel("Durations")
        plt.title(f'Q-Learning ({self.env_id})')
        plt.show()