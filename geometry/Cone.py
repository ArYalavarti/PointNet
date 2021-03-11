import open3d as o3d
import nn.hyperparameters as hp
import numpy as np

from abc import ABC
from geometry.PointCloud import _PointCloud


class Cone(_PointCloud, ABC):
    """
    Factory object for creating point clouds of cone geometries of max height
    hp.MAX_LENGTH and max radius 0.5*hp.MAX_LENGTH
    """

    def __init__(self):
        self.L = 2 * hp.MAX_LENGTH
        self.mesh = o3d.geometry.TriangleMesh.create_cone

    def get_mesh(self):
        l = np.random.randint(1, self.L)
        return self.mesh(radius=0.5*l, height=l)
