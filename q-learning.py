"""
Implementation of the model-free, off-policy, tabular algorithm, Q-learning, using gymnasium environments.
"""
import gymnasium as gym
from collections import defaultdict
import random
import math
import matplotlib.pyplot as plt
from itertools import count

# Hyperparameters
SEED = 42 
EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 5e-5
LR_MAX = 1
LR_MIN = 0.1
LR_DECAY = 1e-4
GAMMA = 0.99
NUM_EPISODES = 10000
EVAL_EPISODES = 10
ENVIRONMENT = "Taxi-v3" # or "FrozenLake-v1" // "CliffWalking-v1"
curr_episode = 0
random.seed(42)

class Agent:
    def __init__(self, env, eps_max, eps_min, eps_decay, lr_max, lr_min, lr_decay, gamma):
        self.env = env
        self.action_space = self.env.action_space
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.gamma = gamma
        # When a new state is encountered, intialize all actions in that state with 0 q-value.
        self.q_table = defaultdict(lambda: {k:0 for k in range(self.action_space.n)})
    
    def action(self, state, eval=False):
        #  Epsilon-Greedy
        sample = random.random()
        eps = max(self.eps_min, self.eps_max * math.exp(-self.eps_decay * curr_episode)) if not eval else 0
        if sample < eps:
            # Sample a random action with probability epsilon
            action = self.action_space.sample()
        else:
            # Choose the action corresponding with max Q-value with probability 1-epsilon.
            action = max(self.q_table[state], key=self.q_table[state].get)
        return action

    def update(self, state, action, reward, next_state, terminated):
        lr = max(self.lr_min, self.lr_max * math.exp(-self.lr_decay * curr_episode))
        current_q_value = self.q_table[state][action]
        if terminated:
            # If the state is terminal, the TD target (future estimated value from being in this state and taking this action) is only the reward.
            td_target = reward
        else:
            optimal_future_value = max(self.q_table[next_state].values())
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
            action = agent.action(state, eval=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state=next_state
            done = terminated or truncated

def training_loop():
    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    agent = Agent(env=env, eps_max=EPSILON_MAX, eps_min=EPSILON_MIN, eps_decay=EPSILON_DECAY, lr_max=LR_MAX, lr_min=LR_MIN, lr_decay=LR_DECAY, gamma=GAMMA)
    episodes = []
    steps_per_episode = []
    for episode in range(NUM_EPISODES):
        global curr_episode
        curr_episode = episode
        state, _  = env.reset(seed=SEED)
        for i in count():
            action = agent.action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state

            if terminated or truncated:
                episodes.append(episode)
                steps_per_episode.append(i)
                break

    plot(episodes, steps_per_episode)
    env.close()
    eval_agent(agent)

if __name__ == "__main__":
    training_loop()