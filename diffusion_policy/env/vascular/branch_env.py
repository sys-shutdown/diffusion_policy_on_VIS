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
from queue import Queue
from OpenGL.GL import *
from OpenGL.GLU import *

from diffusion_policy.env.vascular.branchToolbox import getReward, startCmd, get_ircontroller_state
from diffusion_policy.env.vascular.branchTestScene import createScene

def dfs(cur, goalIdx, path, graph, visited,result):
            if(goalIdx == cur):
                result += path
                return
            for neighbor in graph[cur]:
                if not visited[neighbor]:
                    path += [neighbor]
                    visited[neighbor] = True
                    dfs(neighbor,goalIdx,path,graph,visited,result)
                    visited[neighbor] = False
                    path.pop()


class branchEnv(gym.Env):
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array','dummy']}
    DEFAULT_CONFIG = {"scene": "VIS",
                      "eval": False,
                      "deterministic": True,
                      #"source": [[300, 150, -300],[300, 150, 300]],
                      "source": [[300, 100, 0]],
                      "target": [[0, 100, 0]],
                      'goalPos':None,
                      "goalDir":None,
                      "rotY": 0,
                      "rotZ": 0,
                      "insertion": 0,
                      "start_node": None,
                      "scale_factor": 1,
                      "dt": 0.01,
                      "timer_limit": 80,
                      "timeout": 50,
                      "display_size": (300, 300),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/VIS",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 1000,
                      "distThreshold":100,
                      "time_before_start": 100,
                      "scale": 10,
                      "rotation": [0.0, 0.0, 0.0],
                      "translation": [0.0, 0.0, 0.0],
                      "startNode": 0,
                      "ryRange":[-1,1],
                      "rzRange":[-1,1],
                      "rotationRange":[-3.14,3.14],
                      "insertRange":[0,1],
                      "orthoScale":0.25,
                      "render_mode":"rgb_array",
                      "vessel_model":"YTube.stl",
                      }

    def __init__(self,config=None,randInit=False):
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.randInit = randInit
        self._seed = None
        self.goalPath = None
        self.randShift = None
        self.seed()
        if config is not None:
            self.config.update(config)
        self.render_mode = self.config["render_mode"]
        self.transScale = tS = 1.0
        self.rotScale = rS = 0.5
        self.action_space = spaces.Box(
            low=np.array([-tS,-rS], dtype=np.float64),
            high=np.array([tS,rS], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )
        self.imgQue = Queue(maxsize=32)
        # self.traj = np.zeros((32,3))
        self.observation_space = spaces.Dict({
            'image':spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,self.config["display_size"][0],self.config["display_size"][1]),
                    dtype=np.float32
                ),
            # 'goalCond':spaces.Box(
            #         low=0,
            #         high=1,
            #         shape=(3,self.config["display_size"][0],self.config["display_size"][1]),
            #         dtype=np.float32
            #     ),
            'controllerState':spaces.Box(
                    low=np.array([0,-20], dtype=np.float64),
                    high=np.array([400,20], dtype=np.float64),
                    shape=(2,),
                    dtype=np.float64
            ),  
            'prompt':spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.float32
                ),
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
        image, prompt = self._render_frame(mode)
        controllerState = np.array(get_ircontroller_state(self.root.InstrumentCombined,1))
        obs = {
            'image':image,
            'controllerState':controllerState,
            'prompt':prompt,
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

            glViewport(self.surface_size[0], 0, self.surface_size[0], self.surface_size[1])
            # glColor3f(0.0, 1.0, 0.0)
            # glLineWidth(4.0)
            # glBegin(GL_LINES)
            # vert0 = self.config["nodeVertices"][self.goalPath[0]]+self.randShift[0]
            # for i in range(len(self.goalPath)-1):
            #     node1 =  self.goalPath[i+1]
            #     vert1 = self.config["nodeVertices"][node1]+self.randShift[i+1]
            #     glVertex3f(*vert0)
            #     glVertex3f(*vert1)
            #     vert0 = vert1.copy()
            # glEnd()

            glColor3f(1.0, 1.0, 0.0)
            glPointSize(10.0)
            modelViewMatrix = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64)
            projectionMatrix = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64)
            viewport = np.array(glGetIntegerv(GL_VIEWPORT), dtype=np.int32)
            glBegin(GL_POINTS)
            # self.traj[:-1] = self.traj[1:]
            # self.traj[-1] = self.root.InstrumentCombined.VisuGuide.Quads.position[-1].copy()
            # for vert in self.traj:
            #     glVertex3f(*vert)
            vert = self.root.InstrumentCombined.VisuGuide.Quads.position[-1].copy().astype(np.float64)
            screen_coords = gluProject(vert[0], vert[1], vert[2], modelViewMatrix, projectionMatrix, viewport)
            screen_coords = np.array(screen_coords[:-1])
            screen_coords[0] = np.clip((screen_coords[0]-self.surface_size[0])/self.surface_size[0],0,1)
            screen_coords[1] = np.clip((self.surface_size[1]-screen_coords[1])/self.surface_size[1],0,1)
            glVertex3f(*vert)
            glEnd()
            # glLoadIdentity()
            # cameraMVM = self.root.camera2.getOpenGLModelViewMatrix()
            # glMultMatrixd(cameraMVM)
            # glViewport(self.surface_size[0], 0, self.surface_size[0], self.surface_size[1])
            # Sofa.SofaGL.draw(self.root)
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
        visual_layer = image[:,:self.surface_size[0]]
        prompt_layer = image[:,self.surface_size[0]:]
        self.imgQue.put(visual_layer)
        if(self.imgQue.full()):
            visual_layer = self.imgQue.get()
            image = visual_layer
            # image = cv2.addWeighted(visual_layer, 1.0, prompt_layer, 0.5, 0)
        # glfw.swap_buffers(self.screen)
        if mode == "human":
            
            # image1 = np.concatenate([visual_layer[...,0:1],prompt_layer[...,1:2],visual_layer[...,2:]],axis=-1)
            if self.config['eval']:
                image = visual_layer
            image = image[:,:,(2,1,0)]
            cv2.imshow("observation",image)
            cv2.waitKey(10)
            pygame.display.flip()
        
        return visual_layer, screen_coords

    def reset(self):

        # Set a new random goal from the list
        

        if self.randInit:

            rs = np.random.RandomState(seed=self._seed)
            config = dict()
            config["rotY"] = rs.randint(*self.config["ryRange"])
            config["rotZ"] = rs.randint(*self.config["rzRange"])
            config["rotation"] = rs.rand()*(self.config["rotationRange"][1]-self.config["rotationRange"][0])+self.config["rotationRange"][0]
            config["insertion"] = rs.randint(*self.config["insertRange"])
            config["vessel_model"] = "Meshs/" + str(rs.randint(1,100)) + ".stl"
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