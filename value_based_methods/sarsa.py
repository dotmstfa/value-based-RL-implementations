from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from value_based_methods.utils import tabular_epsilon_greedy

class SARSA:
    """On-policy Tabular TD Control.

    Arguments:
    env_id: Name of environment in gymnasium. *Must have discrete action and observation spaces
    episodes_train: Number of episodes to run for training
    episodes_eval: Number of episodes to run for eval
    gamma: Discount factor
    epsilon_max: Maxiumum epsilon value (higher->more initial exploration)
    epsilon_min: Minimum epsilon value
    epsilon_decay: Decay rate for epsilon
    lr_max: Maximum learning rate value
    lr_min: Minimum learning rate value
    lr_decay: Decay rate for learning rate
    **env_kwargs: Additional arguments to be passed to gymnasium.make().

    Returns:
    None
    """
    def __init__(self, env_id, episodes_train, episodes_eval, gamma, epsilon_max, epsilon_min, epsilon_decay, lr_max, lr_min, lr_decay, **env_kwargs):
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
        self.env_kwargs = env_kwargs
        self.episodes_train = episodes_train
        self.episodes_eval = episodes_eval

        try:
            self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        except:
            raise ValueError("Environment must have discrete observation and action spaces.")

    def train(self):
        for episode in range(self.episodes_train):
            epsilon = max(self.epsilon_min, self.epsilon_max*np.exp(-self.epsilon_decay*episode)) # decay epsilon
            lr = max(self.lr_min, self.lr_max*np.exp(-self.lr_decay*episode)) # decay learning rate
            state, _ = self.env.reset()
            action = tabular_epsilon_greedy(self.q_table, state, epsilon, self.action_space.n)
            for i in count():
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = tabular_epsilon_greedy(self.q_table, next_state, epsilon, self.action_space.n)
                self.update_value(state, action, reward, next_state, next_action, terminated, lr) # update q_table value for [state, action]
                state = next_state
                action = next_action
                done = terminated or truncated
                if done:
                    self.episodes.append(episode)
                    self.durations.append(i)
                    break
        self.env.close()

    def update_value(self, state, action, reward, next_state, next_action, terminated, lr):
        next_value = self.q_table[next_state][next_action] # maximum action value in next_state
        current_value = self.q_table[state][action]
        target = reward + (not terminated)*self.gamma*next_value
        self.q_table[state][action] = current_value + lr*(target-current_value)

    def eval(self):
        env = gym.make(self.env_id, render_mode="human", **self.env_kwargs)
        for i in range(self.episodes_eval):
            state, _ = env.reset()
            while True:
                action = np.argmax(self.q_table[state]) # greedy w.r.t q_values
                next_state, _, terminated, truncated, _ = env.step(action)
                state = next_state
                done = terminated or truncated
                if done:
                    break
        env.close()


    def plot(self):
        plt.plot(self.episodes, self.durations)
        plt.xlabel("Episodes")
        plt.ylabel("Durations")
        plt.title(f'SARSA ({self.env_id})')
        plt.show()