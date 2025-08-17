# Q-learning - Off-policy TD control.
import numpy as np
from itertools import count
from utils.utils import tabular_epsilon_greedy


def q_learning_update(q_table, state, action, reward, next_state, terminated, gamma, lr):
    current_value = q_table[state][action]
    if terminated:
        target = reward
    else:
        optimal_value = np.max(q_table[next_state])
        target = reward + gamma*optimal_value
    q_table[state][action] += lr*(target-current_value)
    return q_table


def q_learning_training(gym, environment, num_episodes, gamma=0.99, epsilon_min=0.05, epsilon_max=1, epsilon_decay=5e-5, lr_min=0.05, lr_max=1, lr_decay=5e-5):
    env = gym.make(environment, render_mode="rgb_array")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    episodes = []
    durations = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        epsilon = max(epsilon_min, epsilon_max*np.exp(-epsilon_decay*episode))
        lr = max(lr_min, lr_max*np.exp(-lr_decay*episode))
        for i in count():
            action = tabular_epsilon_greedy(q_table, state, epsilon, env.action_space.n)
            next_state, reward, terminated, truncated, _ = env.step(action)
            q_table = q_learning_update(q_table, state, action, reward, next_state, terminated, gamma, lr)
            state = next_state
            done = terminated or truncated
            if done:
                episodes.append(episode)
                durations.append(i)
                break
    env.close()
    return q_table, episodes, durations