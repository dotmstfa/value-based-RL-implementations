import numpy as np

def tabular_greedy(q_table, state):
    return np.argmax(q_table[state])

def tabular_epsilon_greedy(q_table, state, epsilon, action_space):
    sample = np.random.random()
    if sample < epsilon:
        action = np.random.randint(0, action_space)
    else:
        action = tabular_greedy(q_table, state)
    return action