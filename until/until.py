import numpy as np

import random
import numpy as np
import os
import pickle


class ReplayMemory:
    """
    a simple replay buffer, suppose the buffer size is big enough to contain all the experiment
    """
    def __init__(self, seed, capacity=1e6):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.start_idx = [0]

    def push(self, state, action, reward, next_state, done):
        assert len(self.buffer) < self.capacity
        self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def finish_path(self, last_val=0):
        """
        Call the function when the episode is finished,
        1. update the beginning of index of trajectory
        2. Todo update discount cumsum
        """
        self.start_idx.append(self.position)

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


def reward_table(env):
    """"
    env: an environment,
    output: the reward table of the environment
    """
    obs_dim = env.observation_space
    action_dim = env.action_dim
    table = np.zeros((obs_dim + [action_dim]))

    for i in range(obs_dim[0]):
        for j in range(obs_dim[1]):
            for k in range(action_dim):
                table[i][j][k] = env.get_instant_reward([i, j], k)

    return table


