import open3d as o3d
import nn.hyperparameters as hp
import numpy as np

from abc import ABC
from geometry.PointCloud import _PointCloud


class Cylinder(_PointCloud, ABC):
    """
    Factory object for creating point clouds of cylinder geometries of max
    height hp.MAX_LENGTH and max radius 0.5*hp.MAX_LENGTH centered at (0, 0, 0)
    """

    def __init__(self):
        self.L = 2 * hp.MAX_LENGTH
        self.mesh = o3d.geometry.TriangleMesh.create_cylinder

    def get_mesh(self):
        l = np.random.randint(1, self.L)
        return self.mesh(radius=0.5*l, height=l)
