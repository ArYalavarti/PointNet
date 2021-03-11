import open3d as o3d
import nn.hyperparameters as hp
import numpy as np

from abc import ABC
from geometry.PointCloud import _PointCloud


class Torus(_PointCloud, ABC):
    """
    Factory object for creating point clouds of torus geometries of max
    torus_radius 0.75 * hp.MAX_RADIUS and max tube_radius 0.25 * hp.MAX_RADIUS
    centered at (0, 0, 0)
    """

    def __init__(self):
        self.R = hp.MAX_RADIUS
        self.mesh = o3d.geometry.TriangleMesh.create_torus

    def get_mesh(self):
        r = np.random.randint(1, self.R)
        return self.mesh(torus_radius=0.75*r, tube_radius=0.25*r)
