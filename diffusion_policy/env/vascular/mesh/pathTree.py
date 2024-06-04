import numpy as np

class segGraph:
    def __init__(self, vertices, lines):
        self.vertices = vertices
        self.lines = lines
        self.vertGroup = dict()
        self.graph = dict()
    
    def buildGraph(self):
        label = 0 
        

class segNode:
    def __init__(self, label):
        self.segment = []
        self.label = label

if __name__ == "__main__":
    import open3d as o3d
    filePath = 'diffusion_policy/env/vascular/mesh/branchSkeleton.obj'

    with open(filePath) as file:
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

    line_set = o3d.geometry.LineSet(
        points= o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([line_set])