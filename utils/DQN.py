"""
This is the code modified from pytorch tutorial to demonstrate the
Q function approximation using NN
"""

import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN_Agent:
    def __init__(self, env, buffer: ReplayMemory, Q_net: DQN, Q_target_net: DQN, batch_size=128,
                 gamma=0.9, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=1000,
                 TAU=0.005, LR=1e-3, device="cpu"):
        self.env = env
        self.policy_net = Q_net
        self.target_net = Q_target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = buffer

        self.batch_size, self.gamma, self.EPS_START = batch_size, gamma, EPS_START
        self.EPS_END, self.EPS_DECAY, self.TAU, self.LR = EPS_END, EPS_DECAY, TAU, LR

        self.total_step = 0
        self.device = device

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def select_action(self, state):
        self.total_step += 1
        # input a batch of state
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.total_step / self.EPS_DECAY)
        self.total_step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # for the random explore stage, only need one sample
            return torch.tensor([[self.env.sample_action()]],
                                device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Q_max(s_t+1)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + \
                                         target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, path=None):
        if not os.path.exists('checkpoints/dqn/model'):
            os.makedirs('checkpoints/dqn/model')
        if path is None:
            path = "checkpoints/dqn/model/model.pt"
        else:
            path = "checkpoints/dqn/model" + path
        print('Saving buffer to {}'.format(path))
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        print("load trained Q net")
        self.policy_net.load_state_dict(torch.load(path))