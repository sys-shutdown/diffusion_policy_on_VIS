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
from queue import Queue

from diffusion_policy.env.vascular.branchToolbox import getReward, startCmd, get_ircontroller_state
from diffusion_policy.env.vascular.VesselScene import createScene

def dfs(cur, goalIdx, path, graph, visited,result):
            if(goalIdx == cur):
                result.append(path.copy())
                return
            if (cur not in graph):
                return
            for neighbor in graph[cur]:
                if not visited[neighbor]:
                    path += [neighbor]
                    visited[neighbor] = True
                    dfs(neighbor,goalIdx,path,graph,visited,result)
                    visited[neighbor] = False
                    path.pop()

class VesselEnv(gym.Env):
    branchModelPath = 'diffusion_policy/env/vascular/mesh/VesselSkeleton.obj'
    with open(branchModelPath) as file:
        points = []
        lines = []
        line = file.readline()
        while line:
            strs = line.split(" ")
            if strs[0] == "v":
                points.append([float(strs[1]),float(strs[2]),float(strs[3])])
            if strs[0] == "l":
                lines.append([int(strs[1])-1,int(strs[2])-1])
            line = file.readline()
    goalList = list(range(len(points)))[1:] 
    graph = dict()
    for line in lines:
        if line[0] in graph:
            graph[line[0]] += [line[1]]
        else:
            graph[line[0]] = [line[1]]
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array','dummy']}
    DEFAULT_CONFIG = {"scene": "VIS",
                      "eval": False,
                      "deterministic": True,
                      "source": [[360, 220, 120]],
                      "target": [[-60, 220, 0]],
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
                      "display_size": (1800, 1800),
                      "obs_size": (300,300),
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
                      "nodeVertices": points,
                      "nodeLines": lines,
                      "nodeGraph": graph,
                      "startNode": 0,
                      "goalList": goalList,
                      "ryRange":[-15,15],
                      "rzRange":[-15,15],
                      "insertRange":[0,30],
                      "orthoScale":0.15,
                      "render_mode":"rgb_array",
                      }

    def __init__(self,config=None,randInit=True):
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.randInit = randInit
        self._seed = None
        self.seed()
        if config is not None:
            self.config.update(config)
        self.render_mode = self.config["render_mode"]
        self.transScale = tS = 4.0
        self.rotScale = rS = 4.0
        self.action_space = spaces.Box(
            low=np.array([-tS,-rS], dtype=np.float64),
            high=np.array([tS,rS], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )
        self.observation_space = spaces.Dict({
            'image':spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,self.config["obs_size"][0],self.config["obs_size"][1]),
                    dtype=np.float32
                ),
            # 'prompt':spaces.Box(
            #         low=0,
            #         high=1,
            #         shape=(2,),
            #         dtype=np.float32
            #     ), 
            'action':self.action_space,  
        })
        self.screen = None
        self.render_cache = None
        self.root = None
        self.surface_size = self.config['display_size']
        self.zFar = self.config['zFar']
        self.goalPath = None
        self.randShift = None
        self.last_action = np.zeros((2,),dtype=np.float64)

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
        # controllerState = np.array(get_ircontroller_state(self.root.InstrumentCombined,1))
        obs = {
            'image':image,
            'action':self.last_action,
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
        self.last_action = action
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
        
        glViewport(0, 0, self.surface_size[0], self.surface_size[1]) # 画面大小
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
            glMultMatrixd(cameraMVM) # 调整相机视角
            Sofa.SofaGL.draw(self.root) # SOFA的渲染函数
            glViewport(self.surface_size[0], 0, self.surface_size[0], self.surface_size[1])
            glColor3f(0.0, 1.0, 0.0)
            glLineWidth(4.0)
            glBegin(GL_LINES)
            vert0 = self.config["nodeVertices"][self.goalPath[0]]+self.randShift[0]
            for i in range(len(self.goalPath)-1):
                node1 =  self.goalPath[i+1]
                vert1 = self.config["nodeVertices"][node1]+self.randShift[i+1]
                glVertex3f(*vert0)
                glVertex3f(*vert1)
                vert0 = vert1.copy()
            glEnd()
            modelViewMatrix = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64)
            projectionMatrix = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64)
            viewport = np.array(glGetIntegerv(GL_VIEWPORT), dtype=np.int32)
            vert = self.root.InstrumentCombined.VisuGuide.Quads.position[-1].copy().astype(np.float64)
            screen_coords = gluProject(vert[0], vert[1], vert[2], modelViewMatrix, projectionMatrix, viewport)
            screen_coords = np.array(screen_coords[:-1])
            screen_coords[0] = screen_coords[0]-self.surface_size[0]
            screen_coords[1] = self.surface_size[1]-screen_coords[1]

            # glViewport(self.surface_size[0], 0, self.surface_size[0], self.surface_size[1])
            # glColor3f(1.0, 1.0, 0.0)
            # glPointSize(10.0)
            # glBegin(GL_POINTS)
            # glVertex3f(*vert)
            # glEnd()
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
        img = np.flipud(image)
        visual_layer = img[:,:self.surface_size[0]]
        prompt_layer = img[:,self.surface_size[0]:]
        img = cv2.addWeighted(visual_layer, 1.0, prompt_layer, 0.5, 0)
        image = img[int(screen_coords[1]-self.config['obs_size'][0]/2):int(screen_coords[1]+self.config['obs_size'][0]/2),int(screen_coords[0]-self.config['obs_size'][1]/2):int(screen_coords[0]+self.config['obs_size'][1]/2)]
        # image = visual_layer
        # glfw.swap_buffers(self.screen)
        if mode == "human":
            
            # image1 = np.concatenate([visual_layer[...,0:1],prompt_layer[...,1:2],visual_layer[...,2:]],axis=-1)
            image = image[:,:,(2,1,0)]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = 255 - image
            cv2.imshow("observation",image)
            cv2.waitKey(10)
            pygame.display.flip()
        
        return image

    def reset(self):

        # Set a new random goal from the list
        if self.randInit:
            rs = np.random.RandomState(seed=self._seed)
            config = dict()
            goal_idx = self.config["goalList"][rs.randint(len(self.config["goalList"]))]
            # goal_idx = 53
            visited = [False for i in range(len(self.config["nodeVertices"]))]
            result = []
            path = [self.config["startNode"]]
            dfs(self.config["startNode"],goal_idx,path,self.config["nodeGraph"],visited,result)
            self.goalPath = result[rs.randint(len(result))]
            self.randShift = np.random.randn(len(self.goalPath),3)*0.5
            dirVec = np.array(self.config["nodeVertices"][self.goalPath[-1]])-np.array(self.config["nodeVertices"][self.goalPath[-2]])
            config["goalPos"] = self.config["nodeVertices"][goal_idx]
            config["goalDir"] = dirVec / np.linalg.norm(dirVec)
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
