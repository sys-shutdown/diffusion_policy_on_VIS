from typing import Any,Dict
import os
import sys
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import splib3
import copy
import Sofa
import cv2
import Sofa.SofaGL
import SofaRuntime
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from diffusion_policy.env.vascular.VISToolbox import getReward, startCmd
from diffusion_policy.env.vascular.VISScene import createScene

class VISEnv(gym.Env):
    
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array','dummy']}
    DEFAULT_CONFIG = {"scene": "VIS",
                      "deterministic": True,
                      "source": [[300, 150, -300],[300, 150, 300]],
                      "target": [[0, 150, 0],[0, 150, 0]],
                      'goalPos':[-5.0, 250.0, 50.0],
                      "rotY": 0,
                      "rotZ": 0,
                      "insertion": 0,
                      "start_node": None,
                      "scale_factor": 3,
                      "dt": 0.01,
                      "timer_limit": 80,
                      "timeout": 50,
                      "display_size": (200, 400),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/VIS",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 500,
                      "distThreshold":100,
                      "time_before_start": 100,
                      "scale": 10,
                      "rotation": [0.0, 0.0, 0.0],
                      "translation": [0.0, 0.0, 0.0],
                      "goalList": [[-5.0, 250.0, 50.0]],
                      "ryRange":[-15,15],
                      "rzRange":[-15,15],
                      "insertRange":[0,80],
                      "orthoScale":0.4,
                      "render_mode":"human",
                      }

    def __init__(self,config=None,randInit=False):
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.randInit = randInit
        self._seed = None
        self.seed()
        if config is not None:
            self.config.update(config)
        self.render_mode = self.config["render_mode"]
        self.transScale = tS = 5.0
        self.rotScale = rS = 5.0
        self.action_space = spaces.Box(
            low=np.array([-tS,-rS], dtype=np.float64),
            high=np.array([tS,rS], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )
        
        self.observation_space = spaces.Dict({
            'image1':spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,self.config["display_size"][0],self.config["display_size"][1]),
                    dtype=np.float32
                ),
            'image2':spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,self.config["display_size"][0],self.config["display_size"][1]),
                    dtype=np.float32
                )
        })
        self.screen = None
        self.render_cache = None
        self.root = None
        self.surface_size = self.config['display_size']
        self.zFar = self.config['zFar']

    def init_simulation(self,config,mode="simu_and_visu"):
        root = Sofa.Core.Node("root")
        #SofaRuntime.importPlugin("Sofa.Component")
        createScene(root, config, mode)
        Sofa.Simulation.init(root)

        if 'time_before_start' in config:
            print(">>   Time before start:", config["time_before_start"], "steps. Initialization ...")
            for i in range(config["time_before_start"]):
                Sofa.Simulation.animate(root, config["dt"])
            print(">>   ... Done.")
            # Update Reward and GoalSetter
            root.GoalSetter.update()
            root.Reward.update()
        return root


    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        
    
    def _get_obs(self,mode="rgb_array"):
        if(mode=="dummy"):
            return dict()
        image = self._render_frame(mode)
        obs = {
            'image1':image[:,0:self.surface_size[0],:],
            'image2':image[:,self.surface_size[0]:,:],
        }
        # cv2.imshow("1",obs['image1'])
        # cv2.imshow("2",obs['image2'])
        # cv2.waitKey(10)
        self.render_cache = image
        return obs

    def step(self,action):
        dt = self.config["dt"]*(self.config["scale_factor"]-1)
        startCmd(self.root,action,dt)

        for i in range(self.config["scale_factor"]):
            Sofa.Simulation.animate(self.root, self.config["dt"])

        obs = self._get_obs(self.config["render_mode"])
        done, reward = getReward(self.root)
        info = {}
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache
    
    def _render_frame(self,mode):
        if self.screen is None:
            if mode == "human":
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption('Vascular Intervention Sugery Simulator')
                self.screen = pygame.display.set_mode((self.surface_size[0]*2,self.surface_size[1]), pygame.OPENGL | pygame.DOUBLEBUF)
                glClearColor(0, 0, 0, 1)
            if mode == "rgb_array":
                # glfw.init()
                # self.screen = glfw.create_window(self.surface_size[0]*2,self.surface_size[1],"1",None,None)
                # glfw.make_context_current(self.screen)
                # glClearColor(0, 0, 0, 1)

                self.screen = pygame.display.set_mode((self.surface_size[0]*2,self.surface_size[1]), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        
        glViewport(0, 0, self.surface_size[0], self.surface_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        if self.root:
            Sofa.SofaGL.glewInit()
            Sofa.Simulation.initVisual(self.root)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            #gluPerspective(45, (self.surface_size[0] / self.surface_size[1]), 0.1, self.zFar)
            scale = self.config['orthoScale']
            glOrtho(-self.surface_size[0]*scale,self.surface_size[0]*scale,-self.surface_size[1]*scale,self.surface_size[1]*scale,-self.zFar,self.zFar)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            cameraMVM = self.root.camera1.getOpenGLModelViewMatrix()
            glMultMatrixd(cameraMVM)
            Sofa.SofaGL.draw(self.root)
            glLoadIdentity()
            cameraMVM = self.root.camera2.getOpenGLModelViewMatrix()
            glMultMatrixd(cameraMVM)
            glViewport(self.surface_size[0], 0, self.surface_size[0], self.surface_size[1])
            Sofa.SofaGL.draw(self.root)
        try:
            x, y, width, height = glGetIntegerv(GL_VIEWPORT)
        except:
            width, height = self.surface_size[0], self.surface_size[1]
        buff = glReadPixels(0, 0, width*2, height, GL_RGB, GL_UNSIGNED_BYTE)

        image_array = np.fromstring(buff, np.uint8)
        if image_array.shape != (0,):
            image = image_array.reshape(self.surface_size[1], self.surface_size[0]*2, 3)
        else:
            image = np.zeros((self.surface_size[1], self.surface_size[0]*2, 3))
        image = np.flipud(image)
        
        # glfw.swap_buffers(self.screen)
        if mode == "human":
            # image = image[:,:,(2,1,0)]
            # cv2.imshow("obervation",image)
            # cv2.waitKey(10)
            pygame.display.flip()
        return image

    def reset(self):

        # Set a new random goal from the list
        if self.randInit:
            rs = np.random.RandomState(seed=self._seed)
            config = dict()
            config["goal"] = self.config["goalList"][0]
            config["rotY"] = rs.randint(*self.config["ryRange"])
            config["rotZ"] = rs.randint(*self.config["rzRange"])
            config["insertion"] = rs.randint(*self.config["insertRange"])
            self.config.update(config)


        self.root = self.init_simulation(self.config)
        if self.config["render_mode"] == "dummy":
            return {}
        observation = self._get_obs(self.config["render_mode"])
        return observation

    def configure(self, config):
        """Update the configuration.

        Parameters:
        ----------
            config: Dictionary.
                Elements to be added in the configuration.

        Returns:
        -------
            None.

        """
        self.config.update(config)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()