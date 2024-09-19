import pathlib
import sys
import numpy as np
from branchToolbox import GoalSetter, RewardShaper, StateInitializer
from splib3.animation import AnimationManagerController
from os.path import abspath, dirname
path = dirname(abspath(__file__)) + '/mesh/'
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=False, drawMode="1", showObjectScale=3.0,
                             showColor=[0, 1, 0, 0.5], position=[-60.0,260.0,47.0])
    return goal

def createScene(root,
                config={"source": [[300, 150, -300],[300, 150, 300]],
                        "target": [[0, 150, 0],[0, 150, 0]],
                        "goalPos": None,
                        "rotY": 0,
                        "rotZ": 0,
                        "rotation": 1.57,
                        "insertion": 0,
                        "zFar":4000,
                        "distThreshold":500,
                        "dt": 0.01},
                mode='simu_and_visu'):
    
    # SETUP
    ## Choose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    ## Root Parameters
    root.name = "root"
    root.gravity=[0.0, 0.0, 0.0]
    root.dt = config['dt']

    plugins_list = ["Sofa.Component.AnimationLoop",
                    "Sofa.Component.IO.Mesh",
                    "Sofa.Component.Mapping.Linear",
                    "Sofa.Component.Mapping.NonLinear",
                    "Sofa.Component.LinearSolver.Direct",
                    "Sofa.Component.LinearSolver.Iterative",
                    "Sofa.Component.ODESolver.Backward",
                    "Sofa.Component.Engine.Generate",
                    "Sofa.Component.Mass",
                    "Sofa.Component.MechanicalLoad",
                    "Sofa.Component.SolidMechanics.Spring",
                    "Sofa.Component.Constraint.Projective",
                    "Sofa.Component.Constraint.Lagrangian.Correction",
                    "Sofa.Component.Constraint.Lagrangian.Model",
                    "Sofa.Component.Constraint.Lagrangian.Solver",
                    "Sofa.Component.StateContainer",
                    "Sofa.Component.Topology.Container.Constant",
                    "Sofa.Component.Topology.Container.Dynamic",
                    "Sofa.Component.Topology.Container.Grid",
                    "Sofa.Component.Topology.Mapping",
                    "Sofa.Component.Collision.Detection.Algorithm",
                    "Sofa.Component.Collision.Detection.Intersection",
                    "Sofa.Component.Collision.Response.Contact",
                    "Sofa.Component.Collision.Geometry",
                    "Sofa.Component.Visual",
                    "Sofa.GL.Component.Rendering3D",
                    "Sofa.GL.Component.Shader",
                    "BeamAdapter",
                    ]
    
    plugins = root.addChild('Plugins')
    for name in plugins_list:
        plugins.addObject('RequiredPlugin', name=name, printLog=False)

    root.addObject('VisualStyle', displayFlags='showVisualModels hideMappings hideForceFields')
    root.addObject('DefaultVisualManagerLoop')

    root.addObject('FreeMotionAnimationLoop')
    root.addObject('LCPConstraintSolver', mu=0.1, tolerance=1e-4, maxIt=2000, build_lcp=False)

    root.addObject('CollisionPipeline', depth=6, verbose=True, draw=False)
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('LocalMinDistance', alarmDistance=2, contactDistance=1, angleCone=0.5, coneFactor=0.5)
    root.addObject('DefaultContactManager', name='Response', response='FrictionContactConstraint')

    # SCENE
    # Catheter
    cath = root.addChild('topoLines_cath')
    cath.addObject('RodStraightSection', name="StraightSection", youngModulus="100000", nbEdgesCollis="40", nbEdgesVisu="220", length="200.0")
    cath.addObject('RodSpireSection', name="SpireSection", youngModulus="100000", nbEdgesCollis="20", nbEdgesVisu="80", length="10.0", spireDiameter="5000.0", spireHeight="0.0")
    cath.addObject('WireRestShape', template="Rigid3d", name="catheterRestShape", wireMaterials="@StraightSection @SpireSection")	
    cath.addObject('EdgeSetTopologyContainer', name='meshLinesCath')
    cath.addObject('EdgeSetTopologyModifier', name='Modifier')
    cath.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo', template='Rigid3d')
    cath.addObject('MechanicalObject', template='Rigid3d', name='dofTopo1')

    ## Guide
    guide = root.addChild('topoLines_guide')
    guide.addObject('RodStraightSection', name="StraightSection", youngModulus="10000", nbEdgesCollis="50", nbEdgesVisu="196", length="240.0")
    guide.addObject('RodSpireSection', name="SpireSection", youngModulus="10000", nbEdgesCollis="20", nbEdgesVisu="10", length="30.0", spireDiameter="40", spireHeight="0.0")
    guide.addObject('WireRestShape', template="Rigid3d", name="GuideRestShape", wireMaterials="@StraightSection @SpireSection")
    guide.addObject('EdgeSetTopologyContainer', name='meshLinesGuide')
    guide.addObject('EdgeSetTopologyModifier', name='Modifier')
    guide.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo', template='Rigid3d')
    guide.addObject('MechanicalObject', template='Rigid3d', name='dofTopo2')
	

    ## Combined Instrument
    instrument = root.addChild('InstrumentCombined')
    instrument.addObject('EulerImplicitSolver', rayleighStiffness=0.2, rayleighMass=0.1, printLog=False)
    instrument.addObject('BTDLinearSolver', subpartSolve=True, verification=False, verbose=False)
    instrument.addObject('RegularGridTopology', name='meshLinesCombined', nx=180, ny=1, nz=1, xmin=0.0, xmax=1.0, ymin=0.0, ymax=0.0, zmin=1, zmax=1)
    instrument.addObject('MechanicalObject', template='Rigid3d', name='DOFs', showIndices=False,ry=0,rz=90)
    
    instrument.addObject('WireBeamInterpolation', name='InterpolCatheter', WireRestShape='@../topoLines_cath/catheterRestShape', radius=1.5, printLog=False)
    instrument.addObject('AdaptiveBeamForceFieldAndMass', name='CatheterForceField', interpolation='@InterpolCatheter', massDensity=0.00000155)	
    
    instrument.addObject('WireBeamInterpolation', name='InterpolGuide', WireRestShape='@../topoLines_guide/GuideRestShape', radius=0.9, printLog=False)
    instrument.addObject('AdaptiveBeamForceFieldAndMass', name='GuideForceField', interpolation='@InterpolGuide', massDensity=0.00000155)
    
    # instrument.addObject('BeamAdapterActionController', name='AController', interventionController='@m_ircontroller',writeMode="0",
    #                      timeSteps=[0.04*i for i in range(25)], actions=[1 for i in range(25)]
    #                      )
    
    instrument.addObject('InterventionalRadiologyController', template='Rigid3d', name='m_ircontroller', printLog=False, xtip=[90, 0], step=3, rotationInstrument=[0,config["rotation"]],
                         controlledInstrument=1, startingPos=[0, 0, 0, 0.707, 0.0, 0.0, 0.707], speed=0, instruments=['InterpolCatheter','InterpolGuide'])
    
    

    instrument.addObject('LinearSolverConstraintCorrection', printLog=False, wire_optimization=True)
    instrument.addObject('FixedConstraint', name='FixedConstraint', indices=0)
    instrument.addObject('RestShapeSpringsForceField', points='@m_ircontroller.indexFirstNode', stiffness=1e8, angularStiffness=1e8)
    
    collis = instrument.addChild('Collis', activated=True)
    collis.addObject('EdgeSetTopologyContainer', name='collisEdgeSet')
    collis.addObject('EdgeSetTopologyModifier', name='colliseEdgeModifier')
    collis.addObject('MechanicalObject', name='CollisionDOFs')
    collis.addObject('MultiAdaptiveBeamMapping', name='collisMap', controller='../m_ircontroller', useCurvAbs=True, printLog=False)
    collis.addObject('LineCollisionModel', proximity=0.0, group=1)
    collis.addObject('PointCollisionModel', proximity=0.0, group=1)
    
    cath_visu = instrument.addChild('VisuCatheter', activated=True)
    cath_visu.addObject('MechanicalObject', name='Quads')
    cath_visu.addObject('QuadSetTopologyContainer', name='ContainerCath')
    cath_visu.addObject('QuadSetTopologyModifier', name='Modifier')
    cath_visu.addObject('QuadSetGeometryAlgorithms', name='GeomAlgo', template='Vec3d')
    cath_visu.addObject('Edge2QuadTopologicalMapping', nbPointsOnEachCircle=10, radius=1.5, input='@../../topoLines_cath/meshLinesCath', output='@ContainerCath', flipNormals=True)
    cath_visu.addObject('AdaptiveBeamMapping', name='VisuMapCath', useCurvAbs=True, printLog=False, interpolation='@../InterpolCatheter', input='@../DOFs', output='@Quads', isMechanical=False)
    
    cath_visuOgl = cath_visu.addChild('VisuOgl', activated=True)
    # cath_visuOgl.addObject('OglModel', name='Visual', color=[0.0, 0.0, 1.0], quads='@../ContainerCath.quads')
    cath_visuOgl.addObject('OglModel', name='Visual', color=[0.1, 0.5, 0.9], quads='@../ContainerCath.quads', material='texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20')
    cath_visuOgl.addObject('IdentityMapping', input='@../Quads', output='@Visual')
    
    guide_visu = instrument.addChild('VisuGuide', activated=True)
    guide_visu.addObject('MechanicalObject', name='Quads')
    guide_visu.addObject('QuadSetTopologyContainer', name='ContainerGuide')
    guide_visu.addObject('QuadSetTopologyModifier', name='Modifier')
    guide_visu.addObject('QuadSetGeometryAlgorithms', name='GeomAlgo', template='Vec3d')
    guide_visu.addObject('Edge2QuadTopologicalMapping', nbPointsOnEachCircle=10, radius=0.9, input='@../../topoLines_guide/meshLinesGuide', output='@ContainerGuide', flipNormals=True, listening=True)
    guide_visu.addObject('AdaptiveBeamMapping', name='visuMapGuide', useCurvAbs=True, printLog=False, interpolation='@../InterpolGuide', input='@../DOFs', output='@Quads', isMechanical=False)
			
    guide_visuOgl = guide_visu.addChild('VisuOgl')
    # guide_visuOgl.addObject('OglModel', name='Visual', color=[1.0, 0.0, 1.0], quads='@../ContainerGuide.quads')
    guide_visuOgl.addObject('OglModel', name='Visual', color=[0.9, 0.9, 0.9], material='texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20', quads='@../ContainerGuide.quads')
    guide_visuOgl.addObject('IdentityMapping', input='@../Quads', output='@Visual')
    

    # Collision
    collision = root.addChild('CollisionModel') 
    collision.addObject('MeshSTLLoader', name='meshLoader', filename=path+'YTube.stl', triangulate=True, flipNormals=True)
    collision.addObject('MeshTopology', position='@meshLoader.position', triangles='@meshLoader.triangles')
    collision.addObject('MechanicalObject', name='DOFs1')
    collision.addObject('TriangleCollisionModel', simulated=False, moving=True)
    collision.addObject('LineCollisionModel', simulated=False, moving=False)
    collision.addObject('PointCollisionModel', simulated=False, moving=False)
    collision.addObject('OglModel', name='Visual', src='@meshLoader', color=[1, 0, 0, 0.2])

    # # Goal
    goal = add_goal_node(root)

    # # SofaGym Env Toolbox
    root.addObject(StateInitializer(name="StateInitializer",rootNode=root))
    root.addObject(RewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos'], goalDir=config['goalDir'], distThreshold=config['distThreshold']))
    root.addObject(GoalSetter(name="GoalSetter", rootNode=root, goal=goal, goalPos=config['goalPos']))


    source = config["source"]
    target = config["target"]
    root.addObject("LightManager",name="light")
    spotloc = [source[0][0], source[0][1]+config["zFar"], 0]
    root.addObject("SpotLight", position=spotloc, direction=[0, -np.sign(source[0][1]), 0])
    root.addObject("InteractiveCamera", name="camera1", position=source[0], lookAt=target[0], zFar=config["zFar"])
    root.addObject("InteractiveCamera", name="camera2", position=source[1], lookAt=target[1], zFar=config["zFar"])
    root.addObject(AnimationManagerController(root, name="AnimationManager"))

    return root


if __name__ == "__main__":
    import SofaRuntime
    import Sofa.Gui
    root = Sofa.Core.Node('root')
    createScene(root)
    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init('myscene', 'qt')
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(2000, 1500)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
