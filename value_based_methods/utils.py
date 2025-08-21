import numpy as np
import matplotlib.pyplot as plt


def tabular_epsilon_greedy(q_table, state, epsilon, action_space):
    sample = np.random.random()
    if sample < epsilon:
        action = np.random.randint(0, action_space)
    else:
        action = np.random.choice(np.flatnonzero(q_table[state] == np.max(q_table[state])))
    return action


def tabular_eval(env, q_table, eval_episodes):
    for _ in range(eval_episodes):
        state, _ = env.reset()
        while True:
            action = np.argmax(q_table[state])
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                break
    env.close()


def plot(epsidoes, durations, method):
    plt.plot(epsidoes, durations)
    plt.xlabel("Epsiode")
    plt.ylabel("Duration")
    plt.title(method)
    plt.show()