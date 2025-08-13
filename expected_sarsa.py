"""Implementation of Expected SARSA"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count

SEED = 42 
EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 5e-5
LR_MAX = 1
LR_MIN = 0.1
LR_DECAY = 1e-4
GAMMA = 0.99
NUM_EPISODES = 100000
EVAL_EPISODES = 10
ENVIRONMENT = "FrozenLake-v1" # or "FrozenLake-v1" // "CliffWalking-v1"
np.random.seed(SEED)

class Agent:
    def __init__(self, env, max_eps, min_eps, eps_decay, max_lr, min_lr, lr_decay, gamma):
        self.env = env
        self.actions_n = env.action_space.n
        self.obs_n = env.observation_space.n
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.curr_eps = max_eps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.q_table = np.zeros((self.obs_n, self.actions_n))
    
    def action(self, state, episode, eval=False):
        self.curr_eps = max(self.min_eps, self.max_eps*np.exp(-self.eps_decay * episode)) if not eval else 0
        if np.random.rand() < self.curr_eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action
    
    def update(self, state, action, next_state, reward, terminated, episode):
        lr = max(self.min_lr, self.max_lr * np.exp(-self.lr_decay * episode))
        current_q = self.q_table[state][action]
        if terminated:
            target = reward
        else:
            exp = np.sum([(self.curr_eps/self.actions_n)*q for q in self.q_table[next_state]])
            exp += (1-self.curr_eps) * np.max(self.q_table[next_state])
            target = reward + self.gamma*exp

        self.q_table[state][action] += lr*(target - current_q)

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Expected SARSA")
    plt.show()

def eval_agent(agent):
    env = gym.make(ENVIRONMENT, render_mode="human")
    for episode in range(EVAL_EPISODES):
        state, _ = env.reset(seed=SEED)
        while True:
            action = agent.action(state, episode, eval=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            if done:
                break

def training():
    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    agent= Agent(env=env, max_eps=EPSILON_MAX, min_eps=EPSILON_MIN, eps_decay=EPSILON_DECAY, max_lr=LR_MAX, min_lr=LR_MIN, lr_decay=LR_DECAY, gamma=GAMMA)
    episodes = []
    durations = []
    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=SEED)
        for i in count():
            action = agent.action(state=state, episode=episode)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state=state, action=action, next_state=next_state, reward=reward, terminated=terminated, episode=episode)
            state = next_state
            done = terminated or truncated
            if done:
                episodes.append(episode)
                durations.append(i)
                if episode % 50000 == 0:
                    print(agent.q_table)
                break
    env.close()
    plot(episodes, durations)
    eval_agent(agent)

training()