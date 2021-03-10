import glob
import random
import re
from collections import defaultdict

import tensorflow as tf
import numpy as np

from datetime import datetime

from datasets import PointCloudDataset
from args import parse_args
from nn import Classification as hp, PointNet, PointNetSoftmaxClassification

tf.keras.backend.set_floatx('float64')

if __name__ == '__main__':
    ARGS = parse_args()

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    model = PointNet()

    # Set up tf checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint_path = ARGS.log_dir + f"/checkpoints"
    if ARGS.load_checkpoint:
        timestamp = ARGS.load_checkpoint

    checkpoint_path += f"/{timestamp}"
    train_log_dir = ARGS.log_dir + f"/logs/{timestamp}/train"
    test_log_dir = ARGS.log_dir + f"/logs/{timestamp}/test"

    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_path,
        max_to_keep=50)

    if ARGS.command != 'train' or ARGS.load_checkpoint:
        # Restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print("Restored checkpoint")

    train_data = PointCloudDataset(ARGS.data_dir, val=False, batch_size=hp.BATCH_SIZE)
    test_data = PointCloudDataset(ARGS.data_dir, val=True, batch_size=hp.BATCH_SIZE)

    try:
        with tf.device("/device:" + ARGS.device):
            if ARGS.command == "train":
                train_obj = PointNetSoftmaxClassification(
                    model, train_log_dir, test_log_dir, manager)

                train_obj.train(train_data, test_data, ARGS.epochs,
                                ARGS.init_epoch)

    except RuntimeError as e:
        # Something went wrong should not get here
        print(e)
