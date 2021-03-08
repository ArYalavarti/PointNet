import open3d as o3d
import nn.hyperparameters as hp

from abc import ABC
from geometry.PointCloud import _PointCloud
from viz import visualize_point_cloud


class Torus(_PointCloud, ABC):
    """
    Factory object for creating point clouds of torus geometries of torus_radius
    0.75 * hp.MAX_RADIUS and tube_radius 0.25 * hp.MAX_RADIUS centered at
    (0, 0, 0)
    """

    def __init__(self):
        self.R = hp.MAX_RADIUS
        self.mesh = o3d.geometry.TriangleMesh.create_torus

    def get_mesh(self):
        return self.mesh(torus_radius=0.75*self.R, tube_radius=0.25*self.R)


if __name__ == '__main__':
    p = Torus().build()
    visualize_point_cloud(p)
