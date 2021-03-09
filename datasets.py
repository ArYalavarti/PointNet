import glob
import tensorflow as tf
import numpy as np
import nn.hyperparameters as hp
from ntpath import basename


class PointCloudDataset:
    """
    Class for containing the training and test sets
    """

    def __init__(self, data_dir, val=False):
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
        for file, label in zip(self.file_list, self.labels):
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
        data = data.batch(hp.BATCH_SIZE)

        return data
