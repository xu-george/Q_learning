#%%
import numpy as np
from env.dis_one_joint import ArmEnv
from utils.util import ReplayMemory
from matplotlib import pyplot as plt

#%%
EP_LEN = 50
expert_len = 50  # number of demonstration
env = ArmEnv(ep_len=EP_LEN)
obs_dim = env.observation_space
action_dim = env.action_dim
seed = 112

# load replay buffer -- state, action, reward, next_state, done
expert_buffer = ReplayMemory(seed=seed, capacity=expert_len*EP_LEN)
expert_buffer.load_buffer("../checkpoints/buffer_one_joint_")
len(expert_buffer.start_idx)

#%%
for traj_num in range(len(expert_buffer.start_idx) - 1):
    init_index = expert_buffer.start_idx[traj_num]
    init_state = expert_buffer.buffer[init_index+1][0]
    env.reset(init_state)
    env.render()

    while init_index < expert_buffer.start_idx[traj_num+1]:
        action = expert_buffer.buffer[init_index][1]
        env.step(action)
        env.render()
        init_index += 1
