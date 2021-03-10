import os
import tensorflow as tf

from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.summary import create_file_writer
from tensorflow.keras.losses import KLD


class PointNetAutoencoderGeneration:
    def __init__(self, ae, train_log_dir, test_log_dir, manager):
        self._ae = ae

        self._ae_loss_fn = KLD
        self._manager = manager

        self._train_ae_loss = Mean(name='train_ae_loss')
        self._test_ae_loss = Mean(name='test_ae_loss')

        self._train_ae_loss.reset_states()
        self._test_ae_loss.reset_states()

        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(test_log_dir, exist_ok=True)

        self._train_summary_writer = create_file_writer(train_log_dir)
        self._test_summary_writer = create_file_writer(test_log_dir)

    @tf.function
    def _train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            autoencoded, _ = self._ae(inputs, training=True)
            x = tf.expand_dims(labels, -1)
            a = tf.gather_nd(autoencoded, x, batch_dims=1)
            ae_loss = self._ae_loss_fn(inputs, a)

        gradients = tape.gradient(ae_loss, self._ae.trainable_variables)
        self._ae.optimizer.apply_gradients(zip(gradients, self._ae.trainable_variables))
        self._train_ae_loss(ae_loss)


    @tf.function
    def _test_step(self, inputs, labels):
        autoencoded, encoded = self._ae(inputs, training=True)
        x = tf.expand_dims(labels, -1)
        a = tf.gather_nd(autoencoded, x, batch_dims=1)
        self._test_ae_loss(self._ae_loss_fn(inputs, a))

    def train(self, train_data, test_data, num_epochs, init_epoch=0):
        for ep_idx in range(num_epochs):
            print(f"===== Epoch {ep_idx+init_epoch+1} =====")
            for input_batch, label_batch in train_data.data:
                self._train_step(input_batch, label_batch)

            for input_batch, label_batch in test_data.data:
                self._test_step(input_batch, label_batch)

            self._update_log(ep_idx+init_epoch + 1)

            self._train_ae_loss.reset_states()
            self._test_ae_loss.reset_states()

            self._manager.save()

    def _update_log(self, epoch_idx):
        with self._train_summary_writer.as_default():
            tf.summary.scalar('ae_loss', self._train_ae_loss.result().numpy(),
                              step=epoch_idx)
        with self._test_summary_writer.as_default():
            tf.summary.scalar('ae_loss', self._test_ae_loss.result().numpy(),
                              step=epoch_idx)
