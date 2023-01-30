import numpy as np
from env.dis_one_joint import ArmEnv

EP_LEN = 50
EP_NUM = 50

if __name__ == '__main__':

    env = ArmEnv(ep_len=EP_LEN)

    # show random action
    for epoch in range(EP_NUM):
        env.reset()
        for step in range(EP_LEN):
            act = env.sample_action()
            obs, _, done = env.step(act)
            print(obs)
            env.render()