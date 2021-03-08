import open3d as o3d
import numpy as np


def triangle_areas(v1, v2, v3):
    """
    Returns the area of each triangle represented by a vertex in v1, v2, and v3
    """
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)


def split_triangles(t):
    """
    Splits the provided triangles into three arrays of vertices

    :param t: np.ndarray of shape [x, 3, 3]
    :return: tuple of np.ndarrays ([x, 3], [x, 3], [x, 3])
    """
    return t[:, 0], t[:, 1], t[:, 2]


class _PointCloud:
    """
    Abstract base class for creating point clouds of n points.
    """

    @property
    def N(self):
        """
        The number of points in the randomly sampled point cloud
        """
        return 1000

    def get_mesh(self, *args):
        """
        TO IMPLEMENT
        :return: open3d.geometry.TriangleMesh for a given shape
        """
        raise NotImplementedError

    def center_point_cloud(self, p, **kwargs):
        return p

    def build(self):
        """
        Generates a point cloud of N random points of the geometry

        :return: np.ndarray of shape [N, 3] representing a point cloud for the
        given shape
        """
        mesh = self.get_mesh()
        faces = np.array(mesh.triangles)
        vertices = np.array(mesh.vertices)

        triangles = vertices[faces]
        p = self.sample_triangles(triangles, self.N)
        p = self.center_point_cloud(p)
        return p

    @staticmethod
    def sample_triangles(t, n):
        """
        Samples the provided triangles from a mesh to generate n points that
        form a point cloud on the mesh. Weights triangles by area to sample
        n triangles, then takes a linear combination of the vertices to sample
        a point within each selected triangle.

        :param t: np.ndarray of shape [x, 3, 3], triangles
        :param n: int, number of points in point cloud
        :return: np.ndarray of shape [n, 3], point cloud
        """
        v1, v2, v3 = split_triangles(t)
        areas = triangle_areas(v1, v2, v3)

        p = areas / areas.sum()
        weighted_indices = np.random.choice(range(len(t)), size=n, p=p)

        t = t[weighted_indices]
        v1, v2, v3 = split_triangles(t)

        u = np.random.rand(n, 1)
        v = np.random.rand(n, 1)

        exceed = u + v >= 1
        u[exceed] = 1-u[exceed]
        v[exceed] = 1-v[exceed]

        w = 1 - (u + v)

        res = v1*u + v2*v + v3*w
        return res

