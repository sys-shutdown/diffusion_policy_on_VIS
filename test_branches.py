import numpy as np
import sys
import os
import time
import pygame
import pyspacemouse
import multiprocessing
from diffusion_policy.env.vascular.branch_env import branchEnv


def readInput(action,running):
    success = pyspacemouse.open()
    if success:
        while running.value==1:
            state = pyspacemouse.read()
            action[0] = -state.pitch*2.0
            action[1] = state.yaw*1.0
            time.sleep(0.01)



if __name__ == '__main__':

    action = multiprocessing.Array("d",[0.0,0.0])
    running = multiprocessing.Value("b",1)
    env_name = "branches-v0"
    config = {
        "display_size": (300, 300),
        "orthoScale":0.25,
        "scale_factor": 3,
        "render_mode":"human",
    }
    env = branchEnv(config,randInit=True)

    print("Start env ", env_name)

    env.configure({"render":1})


    p1 = multiprocessing.Process(target=readInput,args=(action,running))
    p1.start()

    
    while True:

        control_start = False
        retry = False
        done = False
        env.seed()
        env.reset()
        total_reward = 0
        while not done:
            # for event in pygame.event.get():
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_r:
            #             # press "R" to retry
            #             retry=True
            #         elif event.key == pygame.K_q:
            #             # press "Q" to exit
            #             exit(0)
            # # handle control flow
            # if retry:
            #     break

            act = np.array([action[0],action[1]])
            # print(act)

            state, reward, done, info = env.step(act)
            print(state['controllerState'])
            total_reward+=reward


    running.value = 0
    p1.join()

    env.close()
    print("... End.")
