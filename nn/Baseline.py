import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, \
    MaxPool1D, Flatten, GaussianNoise
from tensorflow.keras.optimizers import Adam
from .hyperparameters import Classification as hp


class Baseline(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(Baseline, self).__init__()

        self.optimizer = Adam(learning_rate=hp.learning_rate, beta_1=hp.momentum)
        self.num_classes = hp.classes

        self.gaussian_noise = GaussianNoise(0.02)

        self.arch = [
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dense(256, activation="relu"),
            MaxPool1D(pool_size=10),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dense(self.num_classes)
        ]

    @tf.function
    def call(self, x, training=False):
        x = self.gaussian_noise(x, training=training)
        for layer in self.arch:
            x = layer(x, training=training)

        return x
