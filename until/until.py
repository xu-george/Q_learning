import numpy as np


def reward_table(env):
    """"
    env: an environment,
    output: the reward table of the environment
    """
    obs_dim = env.observation_space
    action_dim = env.action_dim
    table = np.zeros((obs_dim[0], obs_dim[1], action_dim))

    for i in range(obs_dim[0]):
        for j in range(obs_dim[1]):
            for k in range(action_dim):
                table[i][j][k] = env.get_instant_reward([i, j], k)

    return table


