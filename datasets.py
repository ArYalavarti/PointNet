import glob
import tensorflow as tf
import numpy as np
from ntpath import basename
from tqdm import tqdm
from geometry import *


class PointCloudDataset:
    """
    Class for containing the training and test sets
    """

    def __init__(self, data_dir, val=False, batch_size=32):
        self.batch_size = batch_size
        if val:
            self.data_dir = data_dir + "/test"
        else:
            self.data_dir = data_dir + "/train"

        # Set up file lists of point clouds
        self.file_list, self.labels = self._point_cloud_file_list()
        self.data_size = len(self.file_list)

        # Set the dataset's generator attribute
        self.data = self._get_data()
        self.classes = sorted(set(self.labels))

    def _point_cloud_file_list(self):
        """
        Generates shuffled lists of filenames and labels for the data in the
        data_dir. Searches for all .npy files of point clouds
        :return: (np.ndarray, np.ndarray)
        """
        files = glob.glob(self.data_dir + "/**/*.npy")
        labels = list(map(lambda x: basename(x).split("_")[0], files))

        indices = np.arange(len(files))
        np.random.shuffle(indices)

        return np.take(files, indices), np.take(labels, indices)

    def _point_cloud_gen(self):
        """
        Generator that yields a point cloud and associated class label
        :return: (np.ndarray, int)
        """
        for file, label in zip(tqdm(self.file_list), self.labels):
            pc = np.load(file)
            yield pc, self.classes.index(label)

    def _get_data(self):
        """
        Returns a tf.data.Dataset that yields (inputs, labels) of shapes
            ((hp.BATCH_SIZE, 1000, 3), (hp.BATCH_SIZE, ))
        """
        data = tf.data.Dataset.from_generator(
            self._point_cloud_gen,
            output_types=(tf.float64, tf.int64),
            output_shapes=((None, 3), ()))
        data = data.batch(self.batch_size)

        return data


if __name__ == '__main__':
    shapes = [Cone(), Cube(), Cylinder(), Sphere(), Torus()]
    for s in shapes:
        name = str(type(s).__name__).lower()
        for i in range(1000):
            p = s.build()
            np.save(f"data/train/{name}/{name}_{str(i).zfill(4)}", p)
        for i in range(100):
            p = s.build()
            np.save(f"data/test/{name}/{name}_{str(i).zfill(4)}", p)
