import open3d as o3d
import nn.hyperparameters as hp

from abc import ABC
from geometry.PointCloud import _PointCloud


class Cone(_PointCloud, ABC):
    """
    Factory object for creating point clouds of cone geometries of height
    hp.MAX_LENGTH and radius 0.5*hp.MAX_LENGTH centered at (0, 0, 0)
    """

    def __init__(self):
        self.L = 2 * hp.MAX_LENGTH
        self.mesh = o3d.geometry.TriangleMesh.create_cone

    def get_mesh(self):
        return self.mesh(radius=0.5*self.L, height=self.L)

    def center_point_cloud(self, p, **kwargs):
        p[:, 2] -= self.L / 2
        return p
