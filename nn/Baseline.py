import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, \
    MaxPool1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from .hyperparameters import Classification as hp


class Baseline(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(Baseline, self).__init__()

        self.optimizer = Adam(learning_rate=hp.learning_rate)
        self.num_classes = hp.classes

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
        for layer in self.arch:
            x = layer(x, training=training)
        return x


class BaselineRNN(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(BaselineRNN, self).__init__()

        self.optimizer = Adam(learning_rate=hp.learning_rate)
        self.num_classes = hp.classes

        self.arch = [
            LSTM(3, return_sequences=True),
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
        for layer in self.arch:
            x = layer(x, training=training)
        return x
