import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from geometry import *
import open3d as o3d
from tqdm import tqdm


def visualize_point_cloud(point_cloud, pc2):
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        # plt.show()

    else:
        pcd = point_cloud

    o3d.visualization.draw_geometries([pcd, pcd2], mesh_show_back_face=True)


def confusion_plot(model, dataset, mode, filename="", cm=None):
    if cm is None:
        print("Generating confusion matrix")
        labels = []
        predictions = []
        for inputs, l in dataset.data:
            probabilities = model(inputs, training=False).numpy()
            predictions.extend(np.argmax(probabilities, axis=1))
            labels.extend(l.numpy())

        cm = tf.math.confusion_matrix(labels, predictions).numpy()
        np.save(filename, cm)

    # normalized_confusion_matrix = cm / (cm.sum(axis=1)[:, np.newaxis])
    sns.heatmap(cm, cmap="Blues", annot=True, fmt='g',
                xticklabels=dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({mode.capitalize()})")
    plt.show()


def generate_new_shapes_test(model):
    shapes = [Cone(), Cube(), Cylinder(), Sphere(), Torus()]
    correct = 0
    total = 0

    for e, s in enumerate(shapes):
        for i in tqdm(range(1000)):
            p = s.build()
            p = np.expand_dims(p, axis=0)
            pred = model(p)[0]
            pred = np.argmax(pred)
            if e == pred:
                correct += 1
            total += 1
    print(correct/total)
