"""
Implementation of the model-free, off-policy, tabular algorithm, Q-learning, using gymnasium environments.
"""
import gymnasium as gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from itertools import count

# Hyperparameters
SEED = 42
PLOT = True # Plot the results
EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 5e-5
LR_MAX = 1
LR_MIN = 0.1
LR_DECAY = 1e-4
GAMMA = 0.99
NUM_EPISODES = 20000
EVAL_AGENT = True # Evaluate the agent on the learned policy
EVAL_EPISODES = 10
ENVIRONMENT = "FrozenLake-v1" # or "FrozenLake-v1" // "CliffWalking-v1"
random.seed(42)

class Agent:
    def __init__(self, env, eps_max, eps_min, eps_decay, lr_max, lr_min, lr_decay, gamma):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
    
    def select_action(self, state, episode=None, eval=False):
        #  Epsilon-Greedy
        sample = random.random()
        eps = max(self.eps_min, self.eps_max * math.exp(-self.eps_decay * episode)) if not eval else 0
        if sample < eps:
            # Sample a random action with probability epsilon
            action = self.action_space.sample()
        else:
            # Choose the action corresponding with max Q-value with probability 1-epsilon.
            action = np.argmax(self.q_table[state])
        return action

    def update_values(self, state, action, reward, next_state, terminated, episode):
        lr = max(self.lr_min, self.lr_max * math.exp(-self.lr_decay * episode))
        current_q_value = self.q_table[state][action]
        if terminated:
            # If the state is terminal, the TD target (future estimated value from being in this state and taking this action) is only the reward.
            td_target = reward
        else:
            optimal_future_value = np.max(self.q_table[next_state])
            td_target = reward + self.gamma*optimal_future_value
        self.q_table[state][action] += lr*(td_target-current_q_value)

def plot(episodes, steps_per_episode):
    plt.plot(episodes, steps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Q-Learning")
    plt.show()

def eval_agent(agent: Agent):
    # During evaluation, act greedy when choosing actions.
    env = gym.make(ENVIRONMENT, render_mode="human")
    for _ in range(EVAL_EPISODES):
        state, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.select_action(state, eval=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state=next_state
            done = terminated or truncated

def training_loop():
    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    agent = Agent(env=env, eps_max=EPSILON_MAX, eps_min=EPSILON_MIN, eps_decay=EPSILON_DECAY, lr_max=LR_MAX, lr_min=LR_MIN, lr_decay=LR_DECAY, gamma=GAMMA)
    episodes = []
    steps_per_episode = []
    for episode in range(NUM_EPISODES):
        state, _  = env.reset(seed=SEED)
        for i in count():
            action = agent.select_action(state, episode)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update_values(state, action, reward, next_state, terminated, episode)
            state = next_state

            if terminated or truncated:
                episodes.append(episode)
                steps_per_episode.append(i)
                break
    if PLOT:
        plot(episodes, steps_per_episode)

    env.close()
    if EVAL_AGENT:
        eval_agent(agent)

if __name__ == "__main__":
    training_loop()