import numpy as np
import sys
import os
import time
import hydra
import torch
import dill
import pygame
import pyspacemouse
import multiprocessing
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.vascular.branch_env import branchEnv
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pytorch_util import dict_apply
import cv2

def readInput(action,running):
    success = pyspacemouse.open()
    if success:
        while running.value==1:
            state = pyspacemouse.read()
            action[0] = -state.pitch*2.0
            action[1] = state.yaw*1.0
            time.sleep(0.01)


def drawPoint(visu,coords,size=3):
    border = visu.shape
    pointRange = [coords[0]-size,coords[0]+size,coords[1]-size,coords[1]+size]
    if pointRange[0] < 0:
        pointRange[0] = 0
    if pointRange[1] > border[0]:
        pointRange[1] = border[0]
    if pointRange[2] < 0:
        pointRange[2] = 0
    if pointRange[3] > border[1]:
        pointRange[3] = border[1]
    color = [255,255,0]
    visu[pointRange[2]:pointRange[3],pointRange[0]:pointRange[1]] = color
    return visu

if __name__ == '__main__':
    checkpoint = "../Data/TrainModels/2024.10.25/15.39.54_train_diffusion_unet_hybrid_branches_image/checkpoints/epoch=0090-test_mean_score=0.000.ckpt"
    output_dir = "../Data/EvalOutPut/branches"
    device = "cuda:1"
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    np.set_printoptions(precision=2)
    action = multiprocessing.Array("d",[0.0,0.0])
    running = multiprocessing.Value("b",1)
    env_name = "branches-v0"
    start_seed = 0
    env = branchEnv(randInit=True)
    print("Start env ", env_name)

    env.configure({"render_mode":"human","eval":True})
    env.reset()

    p1 = multiprocessing.Process(target=readInput,args=(action,running))
    p1.start()

    start_threshold = 0.001 
    seed = 10
    print(f'starting seed {seed}')
    env.seed(seed)
    env.reset()
    env.render()
    total_reward = 0
    step_idx = 0
    obs_dict = dict()
    imgSeq = np.zeros((2,300,300,3),dtype=np.uint8)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # press "R" to retry
                    retry=True
                elif event.key == pygame.K_q:
                    # press "Q" to exit
                    exit(0)
                # elif event.key == pygame.K_p:
        coords = pygame.mouse.get_pos()
        prompt = np.tile(np.array([coords[0]/300,coords[1]/300]),(2,1))
        print(prompt)
        image = np.moveaxis(imgSeq,-1,1)/255
        np_obs_dict = {"image":np.expand_dims(image,axis=0),"prompt":np.expand_dims(prompt,axis=0)}
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))

        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        actions = np_action_dict['action'][0]
        # print(actions)
        for act in actions:
            state, reward, done, info = env.step(act)
            visu = state['image'][:,:,(2,1,0)]
            drawPoint(visu,coords)
            cv2.imshow("observation2",visu)
            cv2.waitKey(10)
            pygame.display.flip()
            imgSeq[0]=imgSeq[1]
            imgSeq[1]=state['image']
        # handle control flow

        # act = np.array([action[0],action[1]])
        
        # state, reward, done, info = env.step(act)
        # imgSeq[0]=imgSeq[1]
        # imgSeq[1]=state['image']

        