import pytest
from value_based_methods.monte_carlo import MonteCarlo

EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 1e-5
LR_MAX = 0.5
LR_MIN = 0.1
LR_DECAY = 5e-5
GAMMA = 0.99
NUM_EPISODES = 100000
EVAL_EPISODES = 3
ENVIRONMENT = "FrozenLake-v1"
POLICY_TYPE = "off-policy"
VISIT_TYPE = "first-visit"

# is_slippey arg for 'FrozenLake' environment
monte_carlo = MonteCarlo(ENVIRONMENT, NUM_EPISODES, EVAL_EPISODES, GAMMA, EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY, LR_MAX, LR_MIN, LR_DECAY, POLICY_TYPE, VISIT_TYPE, is_slippery=False)
monte_carlo.train()
monte_carlo.plot()
monte_carlo.eval()