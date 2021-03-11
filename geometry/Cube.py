import open3d as o3d
import nn.hyperparameters as hp
import numpy as np

from abc import ABC
from geometry.PointCloud import _PointCloud


class Cube(_PointCloud, ABC):
    """
    Factory object for creating point clouds of torus geometries of max height,
    width, and depth hp.MAX_LENGTH
    """
    def __init__(self):
        self.L = 2 * hp.MAX_LENGTH
        self.mesh = o3d.geometry.TriangleMesh.create_box

    def get_mesh(self):
        l = np.random.randint(1, self.L)
        return self.mesh(l, l, l)
