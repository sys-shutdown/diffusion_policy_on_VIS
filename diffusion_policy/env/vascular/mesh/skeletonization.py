import skeletor as sk
import trimesh
mesh = trimesh.load_mesh('/home/lys/data/Code/diffusion_policy/diffusion_policy/env/vascular/mesh/branches.stl')
fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
skel.show(mesh=True)
