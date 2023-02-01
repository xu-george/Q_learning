import numpy as np
from env.dis_one_joint import ArmEnv
from utils.util import ReplayMemory

EP_LEN = 50
EP_NUM = 50
BufferLen = EP_LEN * EP_NUM
seed = 112

if __name__ == '__main__':
    save_buffer = True
    env = ArmEnv(ep_len=EP_LEN)

    buffer = ReplayMemory(seed=seed, capacity=BufferLen)

    # load a well-trained agent
    Q_table = np.load("checkpoints/Q_value_initial_table.npy")

    # show random action
    for i in range(EP_NUM):
        obs = env.reset()
        done = False

        while not done:
            act = np.argmax(Q_table[obs[0], obs[1], :])
            next_obs, reward, done = env.step(act)

            # save MDP chain to replay buffer--state, action, reward, next_state, done
            if save_buffer:
                buffer.push(state=obs, action=act, reward=reward, next_state=next_obs, done=done)

            obs = next_obs
            print(obs, done)
            # env.render()

        # update begin index
        if save_buffer:
            buffer.finish_path()

    # save replay buffer
    if save_buffer:
        buffer.save_buffer("one_joint")
        # check the load of replay buffer
        new_buffer = ReplayMemory(seed=seed, capacity=BufferLen)
        new_buffer.load_buffer("checkpoints/buffer_one_joint_")

        print("the length of buffer %d" % len(new_buffer.buffer))
        print("the length of trajectory %d" % (len(new_buffer.start_idx)-1))