import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, \
    MaxPool1D, Flatten, Conv1D, Reshape
from tensorflow.keras.optimizers import Adam
from .hyperparameters import Classification as hp


class AE(Model):
    def get_config(self):
        pass

    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.optimizer = Adam(learning_rate=hp.ae_learning_rate,
                              beta_1=hp.momentum)
        self.latent_dim = latent_dim

        self.encoder = Sequential([
            Conv1D(filters=64, kernel_size=1, strides=1, activation="relu"),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=1, strides=1, activation="relu"),
            BatchNormalization(),
            Conv1D(filters=self.latent_dim, kernel_size=1, strides=1,
                   activation="relu"),
            BatchNormalization(),
        ])

        self.decoder = Sequential([
            Dense(256, activation="relu"),
            Dense(256, activation="relu"),
            Dense(1000*3, activation="relu"),
            Reshape((-1, 3))
        ])

    @tf.function
    def call(self, inputs, training=False):
        encoded = self.encoder(inputs)
        encoded = tf.reduce_max(encoded, axis=1)
        decoded = self.decoder(encoded)
        return decoded, encoded


class PointNet(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(PointNet, self).__init__()

        self.optimizer = Adam(learning_rate=hp.learning_rate, beta_1=hp.momentum)
        self.num_classes = hp.classes

        self.arch = [
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dense(self.num_classes)
        ]

    @tf.function
    def call(self, x, training=False):
        for layer in self.arch:
            x = layer(x, training=training)

        return x
