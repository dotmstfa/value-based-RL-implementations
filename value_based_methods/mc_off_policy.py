"""
Implementation of Monte Carlo control algorithm. This is an off-policy implementation, meaning the behaviour policy and target policy are different.
Weighted importance sampling is used.
"""

import gymnasium as gym
from collections import defaultdict, deque, namedtuple
from itertools import count
import random
import math
import matplotlib.pyplot as plt

ENVIRONMENT = "FrozenLake-v1"
GAMMA = 0.99
MIN_EPS = 0.05
MAX_EPS = 1
EPS_DECAY = 3e-5
VISIT_TYPE = "every_visit"
EPISODES = 150000
EVAL_EPISODES = 10
SEED = 42
EVAL_AGENT = True

class Agent:
    def __init__(self, env, gamma, min_eps, max_eps, eps_decay, visit_type='first_visit'):
        self.env = env
        self.action_space = env.action_space
        self.gamma = gamma
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_decay = eps_decay
        self.visit_type = visit_type
        self.q_table = defaultdict(lambda: {k:0 for k in range(self.action_space.n)})
        self.epsiode_visits = defaultdict(lambda: {k:0 for k in range(self.action_space.n)}) # Intra-episode visits to a (state, action) pair
        self.visits = defaultdict(lambda: {k:0 for k in range(self.action_space.n)}) # Total vists to a (state, action) pair
        self.weight_sum = defaultdict(lambda: {k: 0 for k in range(self.action_space.n)}) # C(s,a) -> Weight accumalted for weighted importance sampling
        self.trajectory = deque([], maxlen=None) 
        self.step = namedtuple('Step', 'state action reward episode_visit visit prob')
    
    def action(self, state, episode, eval=False):
        sample = random.random()
        eps = max(self.min_eps, self.max_eps*math.exp(-self.eps_decay*episode)) if not eval else 0
        max_action = max(self.q_table[state], key=self.q_table[state].get)
        if sample < eps:
            action = self.action_space.sample()
        else:
            action = max_action

        if eval:
            return action
        
        if action == max_action:
            prob = (1-eps) + (eps / self.action_space.n)
        else:
            prob = eps / self.action_space.n
        return action, prob

    def update_trajectory(self, step):
        self.trajectory.append(step)

    def update_values(self):
        # Starting at the last step in the trajectory, propogate the return value to earlier steps, and update the action-values 
        # using either the first-visit or every-visit method.
        if self.visit_type == "every_visit":
            curr_return = 0
            w = 1
            while len(self.trajectory) != 0:
                if w > 0:
                    step = self.trajectory.pop()
                    current_value = self.q_table[step.state][step.action]
                    curr_return = step.reward + self.gamma*curr_return
                    self.weight_sum[step.state][step.action] += w
                    self.q_table[step.state][step.action] += (w / self.weight_sum[step.state][step.action])*(curr_return-current_value)
                    max_action = max(self.q_table[step.state], key=self.q_table[step.state].get)
                    if step.action == max_action:
                        w = w*(1 / step.prob)
                    else:
                        break

        elif self.visit_type == "first_visit":
            curr_return = 0
            w = 1
            while len(self.trajectory) != 0:
                if w > 0:
                    step = self.trajectory.pop()
                    current_value = self.q_table[step.state][step.action]
                    curr_return = step.reward + self.gamma*curr_return
                    if step.episode_visit == 1:
                        self.weight_sum[step.state][step.action] += w
                        self.q_table[step.state][step.action] += (w / self.weight_sum[step.state][step.action])*(curr_return-current_value)
                        max_action = max(self.q_table[step.state], key=self.q_table[step.state].get)
                        if step.action == max_action:
                            w = w*(1 / step.prob)
                        else:
                            break
        else:
            raise 'Invalid Visit Type, choose either "first_visit" or "every_visit".'

def training():
    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    agent = Agent(env=env, gamma=GAMMA, min_eps=MIN_EPS, max_eps=MAX_EPS, eps_decay=EPS_DECAY, visit_type=VISIT_TYPE)
    episodes = []
    episode_durations = []
    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED)
        agent.epsiode_visits.clear() # Reset the visits for the episode
        for i in count():
            action, prob = agent.action(state, episode)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.epsiode_visits[state][action] += 1
            if (VISIT_TYPE == "first_visit") and (agent.epsiode_visits[state][action] == 1):
                agent.visits[state][action] += 1 # only increment the total visit counter for the state-action pair if its a first-visit in an episode.
            elif (VISIT_TYPE == "every_visit"):
                agent.visits[state][action] += 1
            step = agent.step(state=state, action=action, reward=reward, episode_visit=agent.epsiode_visits[state][action], visit=agent.visits[state][action], prob=prob)
            agent.update_trajectory(step)
            state = next_state
            done = terminated or truncated
            if done:
                episodes.append(episode)
                episode_durations.append(i)
                break
        agent.update_values() # Monte-Carlo update once episode is finished.
    env.close()
    plot(episodes, episode_durations)
    if EVAL_AGENT:
        evaluation(agent)

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    if VISIT_TYPE == "first_visit":
        plt.title('Monte-Carlo Off-policy (First-visit)')
    elif VISIT_TYPE == "every_visit":
        plt.title('Monte-Carlo Off-policy (Every-visit)')
    plt.show()

def evaluation(agent):
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
        
if __name__ == "__main__":
    random.seed(SEED)
    training()