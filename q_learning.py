import numpy as np
import argparse
from env.dis_one_joint import ArmEnv
from matplotlib import pyplot as plt
from until.until import reward_table

EP_LEN = 50
EP_NUM = 100000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exploration_proba', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--initial', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "main":
    args = parse_args()

    env = ArmEnv(ep_len=EP_LEN)
    obs_dim = env.observation_space
    action_dim = env.action_dim

    if args.initial:
        print("initialize the Q table with instant reward")
        Q_table = reward_table(env)
        print("The shape of Q_table is:", Q_table.shape)
        reward_episode = list()
    else:
        Q_table = np.zeros((obs_dim[0], obs_dim[1], action_dim))
        print("The shape of Q_table is:", Q_table.shape)

    exploration_proba = args.exploration_proba
    gamma = args.gamma
    lr = args.lr

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

        reward_episode.append(episode_reward)

    #%% ----------------- save the Q table
    np.save("./models/Q_value_no_table", Q_table)
    np.save("./results/reward_episode", reward_episode)

    # average every 500 step
    averaged_reward = list()
    for i in range(len(reward_episode)):
        if (i+1) % 500 == 0:
            average = np.mean(reward_episode[i-499:i+1])
            averaged_reward.append(average)
    plt.plot(averaged_reward)
    plt.show()

    #%% --------------- evaluate the performance
    trained_Q_table = np.load("models/Q_value_no_table.npy")

    test_eposide = 20

    for i in range(test_eposide):
        obs = env.reset()
        for step in range(EP_LEN):

            act = np.argmax(Q_table[obs[0], obs[1], :])
            obs, _, done = env.step(act)
            print(obs)
            env.render()

