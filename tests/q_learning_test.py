from value_based_methods.q_learning import q_learning_training
from utils.utils import tabular_eval, plot
import gymnasium as gym

EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 1e-5
LR_MAX = 0.5
LR_MIN = 0.1
LR_DECAY = 5e-5
GAMMA = 0.99
NUM_EPISODES = 10000000
EVAL_EPISODES = 10
ENVIRONMENT = "FrozenLake-v1"


q_table, episodes, durations = q_learning_training(
    gym=gym, environment=ENVIRONMENT, num_episodes=NUM_EPISODES, 
    gamma=GAMMA, epsilon_min=EPSILON_MIN, epsilon_max=EPSILON_MAX, 
    epsilon_decay=EPSILON_DECAY, lr_min=LR_MIN, lr_max=LR_MAX, 
    lr_decay=LR_DECAY,)

plot(episodes, durations, "Q-learning")
tabular_eval(gym, q_table, ENVIRONMENT, EVAL_EPISODES)