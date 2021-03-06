import os
import tensorflow as tf

from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.summary import create_file_writer
from tensorflow.keras.losses import KLD


class PointNetSoftmaxClassification:
    """
    This class implements Softmax classification on the point cloud data
    """

    def __init__(self, model, train_log_dir, test_log_dir, manager):
        self._model = model

        self._loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
        self._manager = manager

        self._train_loss = Mean(name='train_loss')
        self._test_loss = Mean(name='test_loss')

        self._train_acc = SparseCategoricalAccuracy(name='train_acc')
        self._test_acc = SparseCategoricalAccuracy(name='test_acc')

        self._train_loss.reset_states()
        self._test_loss.reset_states()

        self._train_acc.reset_states()
        self._test_acc.reset_states()

        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(test_log_dir, exist_ok=True)

        self._train_summary_writer = create_file_writer(train_log_dir)
        self._test_summary_writer = create_file_writer(test_log_dir)

    @tf.function
    def _train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self._model(inputs, training=True)
            loss = self._loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        self._train_loss(loss)

    @tf.function
    def _test_step(self, inputs, labels):
        predictions = self._model(inputs)
        self._test_loss(self._loss_fn(labels, predictions))
        self._test_acc(labels, predictions)

    @tf.function
    def _train_acc_step(self, inputs, labels):
        predictions = self._model(inputs)
        self._train_acc(labels, predictions)

    def train(self, train_data, test_data, num_epochs, init_epoch=0):
        for ep_idx in range(num_epochs):
            print(f"===== Epoch {ep_idx+init_epoch+1} =====")
            for input_batch, label_batch in train_data.data:
                self._train_step(input_batch, label_batch)

            for input_batch, label_batch in train_data.data:
                self._train_acc_step(input_batch, label_batch)

            for input_batch, label_batch in test_data.data:
                self._test_step(input_batch, label_batch)

            self._update_log(ep_idx+init_epoch + 1)

            self._train_loss.reset_states()
            self._train_acc.reset_states()

            self._test_loss.reset_states()
            self._test_acc.reset_states()

            self._manager.save()

    def test(self, train_data, test_data):
        for input_batch, label_batch in train_data.data:
            self._train_acc_step(input_batch, label_batch)

        for input_batch, label_batch in test_data.data:
            self._test_step(input_batch, label_batch)

        print(f"Train: {self._train_acc.result().numpy()}")
        print(f"Test:  {self._test_acc.result().numpy()}")

    def _update_log(self, epoch_idx):
        with self._train_summary_writer.as_default():
            tf.summary.scalar('loss', self._train_loss.result().numpy(),
                              step=epoch_idx)
            tf.summary.scalar('acc', self._train_acc.result().numpy(),
                              step=epoch_idx)
        with self._test_summary_writer.as_default():
            tf.summary.scalar('loss', self._test_loss.result().numpy(),
                              step=epoch_idx)
            tf.summary.scalar('acc', self._test_acc.result().numpy(),
                              step=epoch_idx)
