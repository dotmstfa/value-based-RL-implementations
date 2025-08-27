from itertools import count
from collections import namedtuple, deque, defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from value_based_methods.utils import tabular_epsilon_greedy

class MonteCarlo:
    """Tabular Monte-Carlo control. Off-policy and On-policy methods. First-visit and Every-visit methods.

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
    policy_type: Either "on-policy" or "off-policy". In the off-policy case, the target policy is greedy.
    visit_type: Either "first-visit" or "every-visit" for value estimation.
    **env_kwargs: Additional arguments to be passed to gymnasium.make().

    Returns:
    None
    """
    def __init__(self, env_id, episodes_train, episodes_eval, gamma, epsilon_max, 
                 epsilon_min, epsilon_decay, lr_max, lr_min, lr_decay, 
                 policy_type="on-policy", visit_type="every-visit", **env_kwargs):
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

        self.policy_type = policy_type
        self.visit_type = visit_type
        self.visits = defaultdict(int)
        self.episode_visits = defaultdict(int)
        self.step = namedtuple("Step", "state action reward first_visit visit prob")
        self.trajectory = deque([], maxlen=None)
        self.gamma = gamma
        self.env_kwargs = env_kwargs
        self.episodes_train = episodes_train
        self.episodes_eval = episodes_eval

        if policy_type == "off-policy":
            self.weight_sum = np.zeros((self.observation_space.n, self.action_space.n))

        try:
            self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        except:
            raise ValueError("Environment must have discrete observation and action spaces.")
        
        if (visit_type != "first-visit" and visit_type != "every-visit"):
            raise ValueError("visit_type must be either 'first-visit' or 'every-visit'")
        
        if (policy_type != "on-policy" and policy_type != "off-policy"):
            raise ValueError("policy_type must be 'on-policy' or 'off-policy'")

    def train(self):
        for episode in range(self.episodes_train):
            epsilon = max(self.epsilon_min, self.epsilon_max*np.exp(-self.epsilon_decay*episode)) # decay epsilon
            lr = max(self.lr_min, self.lr_max*np.exp(-self.lr_decay*episode)) # decay learning rate
            self.trajectory.clear() # Clear trajectory
            self.episode_visits = defaultdict(int) # Set all states to not visited
            state, _ = self.env.reset()
            for i in count():
                action = tabular_epsilon_greedy(self.q_table, state, epsilon, self.action_space.n)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if (self.visit_type == "first-visit" and self.episode_visits[state] == 0) or (self.visit_type == "every-visit"):
                    episode_visit = self.episode_visits[state]
                    if episode_visit == 0:
                        first_visit = True
                    else:
                        first_visit = False
                    self.episode_visits[state] = 1
                    self.visits[state] += 1
                    step = self.step(state, action, reward, first_visit, self.visits[state], self.get_prob(state, action, epsilon))
                    self.update_trajectory(step)
                state = next_state
                done = terminated or truncated
                if done:
                    self.episodes.append(episode)
                    self.durations.append(i)
                    break
            if self.policy_type == "on-policy":
                self.on_policy_update()
            else:
                self.off_policy_update()
        self.env.close()

    def update_trajectory(self, step):
        self.trajectory.append(step)

    def get_prob(self, state, action, epsilon):
        optimal_action = np.argmax(self.q_table[state])
        if action == optimal_action:
            prob = (1-epsilon) + (epsilon / self.action_space.n)
        else:
            prob =  (epsilon / self.action_space.n)
        return prob

    def on_policy_update(self):
        curr_return = 0
        while len(self.trajectory) > 0:
            step = self.trajectory.pop()
            curr_return = step.reward + self.gamma*curr_return
            current_value = self.q_table[step.state][step.action]
            if (self.visit_type == "first-visit" and step.first_visit) or (self.visit_type == "every-visit"):
                self.q_table[step.state][step.action] = current_value + (1/step.visit)*(curr_return-current_value)

    def off_policy_update(self):
        curr_return = 0
        weight = 1
        while len(self.trajectory) > 0:
            if weight > 0:
                step = self.trajectory.pop()
                target = step.reward + self.gamma*curr_return
                curr_return = target
                current_value = self.q_table[step.state][step.action]
                if ((self.visit_type == "first-visit") and (step.first_visit)) or (self.visit_type == "every-visit"):
                    self.weight_sum[step.state][step.action] += weight
                    self.q_table[step.state][step.action] = current_value + (weight/self.weight_sum[step.state][step.action])*(curr_return-current_value)
                    if step.action == np.argmax(self.q_table[step.state]):
                        weight = weight * 1/step.prob
                    else:
                        break

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
        plt.title(f'Monte-Carlo ({self.env_id})')
        plt.show()