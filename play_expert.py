import numpy as np
from env.dis_one_joint import ArmEnv

EP_LEN = 50
EP_NUM = 50

if __name__ == '__main__':

    env = ArmEnv(ep_len=EP_LEN)

    # load a well-trained agent
    Q_table = np.load("models/Q_value_table.npy")

    # show random action
    for i in range(EP_NUM):
        obs = env.reset()
        done = False
        while not done:
            act = np.argmax(Q_table[obs[0], obs[1], :])
            obs, _, done = env.step(act)
            print(obs)
            env.render()