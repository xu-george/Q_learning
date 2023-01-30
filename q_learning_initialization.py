import numpy as np
from env.dis_one_joint import ArmEnv
from matplotlib import pyplot as plt
from until.until import reward_table

EP_LEN = 50
EP_NUM = 100000
#%%
env = ArmEnv(ep_len=EP_LEN)
obs_dim = env.observation_space
action_dim = env.action_dim

#%% --------- hyper-paramters -----------
exploration_proba = 0.1
#min_exploration_proba = 0.01
#exploration_decay = 0.0001
# discounted factor
gamma = 0.99
# learning rate
lr = 0.1

#%% Initialize the Q table with zeros
print("initialize the Q table with instant reward")
Q_table = reward_table(env)
print("The shape of Q_table is:", Q_table.shape)
reward_episode = list()

#%% update Q_table
for eposide in range(EP_NUM):
    current_state = env.reset()
    done = False
    episode_reward = 0

    for i in range(EP_LEN):
        if np.random.uniform(0, 1) < exploration_proba:
            action = env.sample_action()
        else:
            action = np.argmax(Q_table[current_state[0], current_state[1], :])

        next_state, reward, done = env.step(action)
        episode_reward += reward

        # update Q table:
        Q_table[current_state[0], current_state[1], action] = (1 - lr) * Q_table[current_state[0], current_state[1], action] + \
                                                               lr * (reward + gamma * max(Q_table[next_state[0], next_state[1], :]))
        if done:
            break

        current_state = next_state
        #exploration_proba = max(min_exploration_proba, np.exp(-exploration_decay * i))

    reward_episode.append(episode_reward)

#%% ----------------- save the Q table
np.save("Q_value_table", Q_table)
np.save("initital_reward_episode", reward_episode)

#%% -----------------plot training process
# average every 500 step
averaged_reward = list()
for i in range(len(reward_episode)):
    if (i+1) % 500 == 0:
        average = np.mean(reward_episode[i-499:i+1])
        averaged_reward.append(average)
plt.plot(averaged_reward)
plt.show()

#%% --------------- evaluate the performance
trained_Q_table = np.load("models/Q_value_table.npy")

test_eposide = 20

for i in range(test_eposide):
    obs = env.reset()
    for step in range(EP_LEN):

        act = np.argmax(Q_table[obs[0], obs[1], :])
        obs, _, done = env.step(act)
        print(obs)
        env.render()

