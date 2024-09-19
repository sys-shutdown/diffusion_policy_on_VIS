import numpy as np
import sys
import os
import time
import pygame
import pyspacemouse
import multiprocessing
from diffusion_policy.env.vascular.branch_env import branchEnv
from diffusion_policy.common.replay_buffer import ReplayBuffer

def readInput(action,running):
    success = pyspacemouse.open()
    if success:
        while running.value==1:
            state = pyspacemouse.read()
            action[0] = -state.pitch*1.0
            action[1] = state.yaw*0.5
            time.sleep(0.01)



if __name__ == '__main__':

    np.set_printoptions(precision=2)
    action = multiprocessing.Array("d",[0.0,0.0])
    running = multiprocessing.Value("b",1)
    env_name = "branches-v0"
    start_seed = 0
    env = branchEnv(randInit=True)

    print("Start env ", env_name)

    env.configure({"render_mode":"human"})
    env.reset()

    p1 = multiprocessing.Process(target=readInput,args=(action,running))
    p1.start()

    output = "../Data/TrainData/branches_demo3.zarr"
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')


    start_threshold = 0.001 
    
    while True:
        episode = list()
        seed = start_seed+replay_buffer.n_episodes
        print(f'starting seed {seed}')
        env.seed(seed)
        control_start = False
        retry = False
        done = False
        env.reset()
        env.render()
        total_reward = 0
        step_idx = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
            # handle control flow
            if retry:
                break

            act = np.array([action[0],action[1]])
            
            if control_start is False:
                if abs(act[0])>start_threshold or abs(act[1])>start_threshold:
                    control_start = True

            
            if control_start:
                state, reward, done, info = env.step(act)
                print(f'step: {step_idx},\t action: {act},\t state: {state["controllerState"]}')
                step_idx+=1
                total_reward+=reward
                data = {
                    'image': state['image'],
                    # 'goalCond': state['goalCond'],
                    'controllerState': state['controllerState'],
                    'action': np.float32(act),
                }
                episode.append(data)

        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')

    running.value = 0
    p1.join()

    env.close()
    print("... End.")
