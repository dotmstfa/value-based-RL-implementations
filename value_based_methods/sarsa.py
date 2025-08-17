# SARSA(0)/one-step SARSA - On-policy TD control.
import numpy as np
from itertools import count
from utils.utils import tabular_greedy, tabular_epsilon_greedy

def sarsa_update(q_table, state, action, reward, next_state, next_action, gamma, lr, terminated):
    current_value = q_table[state][action]
    if terminated:
        target = reward
    else:
        next_value = q_table[next_state][next_action]
        target = reward + gamma*next_value
    q_table[state][action] += lr*(target - current_value)
    return q_table


def sarsa_training(gym, environment, seed, training_episodes, gamma=0.99, eps_min=0.05, eps_max=1, eps_decay=5e-5, lr_min=0.05, lr_max=1, lr_decay=5e-5):
    np.random.seed(seed=seed)
    env = gym.make(environment, render_mode="rgb_array")
    episodes = []
    durations = []
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(training_episodes):
        epsilon = max(eps_min, eps_max * np.exp(-eps_decay*episode)) 
        lr = max(lr_min, lr_max*np.exp(-lr_decay*episode))
        state, _ = env.reset(seed=seed)
        action = tabular_epsilon_greedy(q_table, state, epsilon, env.action_space.n, seed)
        for i in count():
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                next_action = None
            else:
                next_action = tabular_epsilon_greedy(q_table, next_state, epsilon, env.action_space.n, seed)
            q_table = sarsa_update(q_table, state, action, reward, next_state, next_action, gamma, lr, terminated)
            state = next_state
            action = next_action
            done = terminated or truncated
            if done:
                episodes.append(episode)
                durations.append(i)
                break
    env.close()
    return q_table, episodes, durations