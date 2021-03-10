import open3d as o3d
import nn.hyperparameters as hp

from abc import ABC
from geometry.PointCloud import _PointCloud


class Cube(_PointCloud, ABC):
    """
    Factory object for creating point clouds of torus geometries of height,
    width, and depth hp.MAX_LENGTH centered at (0, 0, 0)
    """
    def __init__(self):
        self.L = 2 * hp.MAX_LENGTH
        self.mesh = o3d.geometry.TriangleMesh.create_box

    def get_mesh(self):
        return self.mesh(self.L, self.L, self.L)

    def center_point_cloud(self, p, **kwargs):
        return p - (self.L / 2)
