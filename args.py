import os
import tensorflow as tf

from argparse import ArgumentTypeError, ArgumentParser

gpu_available = tf.config.list_physical_devices("GPU")
print(gpu_available)


def parse_args():
    def valid_dir(directory):
        if os.path.isdir(directory):
            return os.path.normpath(directory)
        else:
            raise ArgumentTypeError("Invalid directory: {}".format(directory))

    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=valid_dir,
        default=os.getcwd() + "/data",
        help="Directory containing point cloud data")

    parser.add_argument(
        "--log-dir",
        default=os.getcwd() + "/log",
        help="Directory to store checkpoint model weights and TensorBoard logs")

    parser.add_argument(
        "--device",
        type=str,
        default="GPU:0" if gpu_available else "CPU:0",
        help="Specify the device of computation eg. CPU:0, GPU:0, GPU:1, ... ")

    parser.add_argument(
        "--load-checkpoint",
        required=False,
        help="Use this flag to resuming training from a previous checkpoint")

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    # Subparser for train command
    tn = subparsers.add_parser(
        "train",
        description="Train a new model")
    tn.set_defaults(command="train")

    tn.add_argument(
        "--epochs",
        type=int, default=10,
        help="Number of epochs to train for")

    tn.add_argument(
        "--init-epoch",
        type=int, default=0,
        help="epoch id to start training at")

    ts = subparsers.add_parser(
        "test",
        description="Test the trained model")
    ts.set_defaults(command="test")

    ev = subparsers.add_parser(
        "evaluate",
        description="Evaluate the trained model")
    ev.set_defaults(command="evaluate")

    ev.add_argument(
        "--mode",
        required=True,
        default="train",
        type=str,
        help="Train or test")

    return parser.parse_args()
