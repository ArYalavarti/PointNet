import open3d as o3d
import nn.hyperparameters as hp

from abc import ABC
from geometry.PointCloud import _PointCloud
from viz import visualize_point_cloud


class Sphere(_PointCloud, ABC):
    """
    Factory object for creating point clouds of sphere geometries of radius
    hp.MAX_RADIUS centered at (0, 0, 0)
    """
    def __init__(self):
        self.R = hp.MAX_RADIUS
        self.mesh = o3d.geometry.TriangleMesh.create_sphere

    def get_mesh(self):
        return self.mesh(self.R)


if __name__ == '__main__':
    p = Sphere().build()
    visualize_point_cloud(p)
