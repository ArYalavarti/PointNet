import tensorflow as tf
import numpy as np

from datetime import datetime

from datasets import PointCloudDataset
from args import parse_args
from nn import Classification as hp, PointNet, PointNetSoftmaxClassification, AE
from eval import confusion_plot, generate_new_shapes_test
from geometry import *

tf.keras.backend.set_floatx('float64')

if __name__ == '__main__':
    ARGS = parse_args()

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    model = PointNet()
    ae = AE(128)

    # Set up tf checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model, ae=ae)

    checkpoint_path = ARGS.log_dir + "/checkpoints"
    if ARGS.load_checkpoint:
        timestamp = ARGS.load_checkpoint

    checkpoint_path += f"/{timestamp}"
    train_log_dir = ARGS.log_dir + f"/logs/{timestamp}/train"
    test_log_dir = ARGS.log_dir + f"/logs/{timestamp}/test"

    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_path,
        max_to_keep=1)

    if ARGS.command != 'train' or ARGS.load_checkpoint:
        # Restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print("Restored checkpoint")

    train_data = PointCloudDataset(ARGS.data_dir, val=False, batch_size=hp.BATCH_SIZE)
    test_data = PointCloudDataset(ARGS.data_dir, val=True, batch_size=hp.BATCH_SIZE)

    train_obj = PointNetSoftmaxClassification(
        model, ae, train_log_dir, test_log_dir, manager)

    try:
        with tf.device("/device:" + ARGS.device):
            if ARGS.command == "train":
                train_obj.train(train_data, test_data, ARGS.epochs,
                                ARGS.init_epoch)

            elif ARGS.command == "test":
                train_obj.test(train_data, test_data)

            elif ARGS.command == 'evaluate':
                confusion = None
                if ARGS.load_confusion:
                    confusion = np.load(ARGS.load_confusion)

                if ARGS.mode == 'train':
                    confusion_plot(model, train_data, ARGS.mode, cm=confusion,
                                   filename="train")
                else:
                    confusion_plot(model, test_data, ARGS.mode, cm=confusion,
                                   filename="test")

    except RuntimeError as e:
        # Something went wrong should not get here
        print(e)
