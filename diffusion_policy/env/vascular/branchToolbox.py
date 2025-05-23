import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

SofaRuntime.importPlugin("Sofa.Component")

class StateInitializer(Sofa.Core.Controller):
    """Initialize the states.

    Methods:
    -------
        __init__: Initialization of all arguments.
        init_state: Randomly initialize the environment state.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.

    """
    def __init__(self, rootNode, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = rootNode
            


    def init_state(self):
        
        pass

class RewardShaper(Sofa.Core.Controller):
    """Compute the reward.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getReward: Compute the reward.
        update: Initialize the value of cost.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.
        goal_pos: coordinates
            The position of the goal.
        effMO: <MechanicalObject>
            The mechanical object of the element to move.
        cost:
            Evolution of the distance between object and goal.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.root = kwargs["rootNode"]
        self.goal_pos = None
        if kwargs["goalPos"] is not None:
            self.goal_pos = np.array(kwargs["goalPos"])
        if kwargs["goalDir"] is not None:
            self.goal_dir = kwargs["goalDir"]
        self.distThreshold = 50
        if kwargs["distThreshold"]:
            self.distThreshold = kwargs["distThreshold"]


        self.init_dist = None
        self.prev_dist = None

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """    
        tip = np.array(self.root.InstrumentCombined.VisuGuide.Quads.position[-1])
        dist_vec = self.goal_pos-tip
        proj_dist = np.dot(dist_vec,self.goal_dir)
        plane_dist = np.sum((dist_vec - proj_dist*self.goal_dir)**2)
        total_dist = np.sqrt(proj_dist**2 + plane_dist)
        # current_dist = np.linalg.norm(np.array(tip)-np.array(self.goal_pos))
        if (total_dist > self.distThreshold):
            reward = 0
        else:
            reward = np.clip((self.distThreshold-proj_dist)/self.distThreshold,0,2)
        return reward

    def update(self):
        """Update function.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        tip = self.root.InstrumentCombined.DOFs.position[-1][:3]
        # self.init_dist = np.linalg.norm(np.array(tip)-np.array(self.goal_pos))
        # self.prev_dist = self.init_dist


class GoalSetter(Sofa.Core.Controller):
    """Compute the goal.

    Methods:
    -------
        __init__: Initialization of all arguments.
        update: Initialize the value of cost.

    Arguments:
    ---------
        goalMO: <MechanicalObject>
            The mechanical object of the goal.
        goalPos: coordinates
            The coordinates of the goal.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]
        self.goal = None
        if kwargs["goal"]:
            self.goal = kwargs["goal"]
        self.goalPos = None
        if kwargs["goalPos"] is not None:
            self.goalPos = kwargs["goalPos"]

    def update(self):
        """Set the position of the goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        new_position = self.goalPos
        with self.goal.GoalMO.position.writeable() as position:
            position[0] = new_position

    def set_mo_pos(self, goal):
        """Modify the goal.

        Not used here.
        """
        pass


def _getGoalPos(root):
    """Get XYZ position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        The position of the goal.
    """
    return root.Goal.GoalMO.position[0]


def getState(root):
    """Compute the state of the environment/agent.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    xtips = []
    rotations = []

    for instrument in range(2):
        xtips.append(root.InstrumentCombined.m_ircontroller.xtip.value[instrument].tolist())
        rotations.append(root.InstrumentCombined.m_ircontroller.rotationInstrument.value[instrument].tolist())

    tip = root.InstrumentCombined.DOFs.position[-1][:3].tolist()

    goal_pos = _getGoalPos(root).tolist()

    state = xtips + rotations + tip + goal_pos

    return state


def getReward(root):
    """Compute the reward using Reward.getReward().

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        done, reward

    """
    reward = root.Reward.getReward()

    if reward >= 1.0:
        return True, 2.0

    return False, 0


def get_ircontroller_state(node, instrument=0):
    """
    Get state (translation, rotation) of th Interventional Radiology Controller
    """
    return [float(node.m_ircontroller.xtip.value[instrument]),
            float(node.m_ircontroller.rotationInstrument.value[instrument])]


def startCmd(root, action, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float
            Duration of the animation.

    Returns:
    ------
        None.

    """
    scale = int(duration/0.01 + 1)
    controlled_instrument, cmd_translation, cmd_rotation = action_to_command(action, scale)
    source = get_ircontroller_state(root.InstrumentCombined, instrument=controlled_instrument)
    target_translation = source[0] + cmd_translation
    target = [target_translation if target_translation > 0 else 0.1, source[1] + cmd_rotation]
    start_cmd(root, root.InstrumentCombined, source, target, duration, controlled_instrument)


def start_cmd(rootNode, IRC_node, source, target, duration, instrument=0):
    def execute_animation(controller, anim_source, anim_target, factor, anim_instrument):
        """
        Execute animation on the IRC to go from source to target
        """
        with controller.xtip.writeable() as xtip:
            xtip[anim_instrument] = anim_source[0] + (anim_target[0] - anim_source[0]) * factor
        if anim_instrument == 1:
            with controller.rotationInstrument.writeable() as rotation:
                rotation[1] = anim_source[1] + (anim_target[1] - anim_source[1]) * factor

    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=execute_animation,
            params={"controller": IRC_node.m_ircontroller,
                    "anim_source": source,
                    "anim_target": target,
                    "anim_instrument": instrument},
            duration=duration, mode="once"))

    return


def action_to_command(action, scale):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).

    Returns:
    -------
        The command (number of the cabl and its displacement).
    """
    # if action == 0:
    #     controlled_instrument = 1
    #     cmd_translation = 2.0 * scale / 2
    #     cmd_rotation = 0.0
    # elif action == 1:
    #     controlled_instrument = 1
    #     cmd_translation = 0.0
    #     cmd_rotation = 1/15 * scale / 2
    # elif action == 2:
    #     controlled_instrument = 1
    #     cmd_translation = 0.0
    #     cmd_rotation = -1/15 * scale / 2
    # elif action == 3:
    #     controlled_instrument = 1
    #     cmd_translation = -0.7 * scale / 2
    #     cmd_rotation = 0.0
    controlled_instrument = 1
    cmd_translation = 0.5 * action[0] * scale
    cmd_rotation = -1/40 * action[1] * scale

    return controlled_instrument, cmd_translation, cmd_rotation


def getPos(root):
    """Retun the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.

    Returns:
    -------
        _: list
            The position(s) of the object(s) of the scene.
    """
    cath_xtip = root.InstrumentCombined.m_ircontroller.xtip.value[0].tolist()
    cath_rotation = root.InstrumentCombined.m_ircontroller.rotationInstrument.value[0].tolist()
    guide_xtip = root.InstrumentCombined.m_ircontroller.xtip.value[1].tolist()
    guide_rotation = root.InstrumentCombined.m_ircontroller.rotationInstrument.value[1].tolist()    

    tip = root.InstrumentCombined.DOFs.position.value.tolist()
    collis = root.InstrumentCombined.Collis.CollisionDOFs.position.value.tolist()
    
    return [cath_xtip, cath_rotation, guide_xtip, guide_rotation, tip, collis]


def setPos(root, pos):
    """Set the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.
        pos: list
            The position(s) of the object(s) of the scene.

    Returns:
    -------
        None.

    Note:
    ----
        Don't forget to init the new value of the position.

    """
    cath_xtip, cath_rotation, guide_xtip, guide_rotation, tip, collis = pos
    
    controller = root.InstrumentCombined.m_ircontroller
    with controller.xtip.writeable() as xtip:
        xtip[0] = np.array(cath_xtip)
        xtip[1] = np.array(guide_xtip)
    
    with controller.rotationInstrument.writeable() as rotation:
        rotation[0] = np.array(cath_rotation)
        rotation[1] = np.array(guide_rotation)

    root.InstrumentCombined.DOFs.position.value = np.array(tip)
    root.InstrumentCombined.Collis.CollisionDOFs.position.value = np.array(collis)
