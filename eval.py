import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from geometry import *
import open3d as o3d
from tqdm import tqdm


def visualize_point_cloud(point_cloud):
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        # plt.show()

    else:
        pcd = point_cloud

    o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)


def confusion_plot2(model, dataset, mode, filename="", cm=None):
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

    sns.heatmap(cm, cmap="Blues", annot=True, fmt='g',
                xticklabels=dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({mode.capitalize()})")
    plt.show()


def generate_new_shapes_test2(model):
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


def confusion_plot(model, ae, dataset, mode, filename="", cm=None):
    if cm is None:
        print("Generating confusion matrix")
        labels = []
        predictions = []
        for inputs, l in dataset.data:
            predictions.extend(model.predict(ae, inputs))
            labels.extend(l.numpy())

        cm = tf.math.confusion_matrix(labels, predictions).numpy()
        np.save(filename, cm)

    sns.heatmap(cm, cmap="Blues", annot=True, fmt='g',
                xticklabels=dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({mode.capitalize()})")
    plt.show()


def generate_new_shapes_test(model, ae):
    shapes = [Cone(), Cube(), Cylinder(), Sphere(), Torus()]
    correct = 0
    total = 0

    # x = np.random.normal(size=(1, 128))
    # y = ae.decode(x)
    # visualize_point_cloud(y.numpy()[0])

    for e, s in enumerate(shapes):
        for i in tqdm(range(1000)):
            p = s.build()
            p = np.expand_dims(p, axis=0)
            pred = model.predict(ae, p)[0]
            if e == pred:
                correct += 1
            total += 1
    print(correct/total)
