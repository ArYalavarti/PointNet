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

    # @tf.function
    def call(self, inputs, training=False):
        encoded = self.encoder(inputs)
        encoded = tf.reduce_max(encoded, axis=1)
        decoded = self.decoder(encoded)
        return decoded


  #
  #
  #
  #       self.encoder = Sequential([
  #
  #
  #
  #
  #       ])
  #
  # def call(self, x):
  #
  #
  #




class PointNet(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(PointNet, self).__init__()

        self.optimizer = Adam(learning_rate=hp.learning_rate, beta_1=hp.momentum)
        self.num_classes = hp.classes

        self.ae = AE(latent_dim=128)

        # Input transformation subnet
        self.t1_mlp = [
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dense(1024, activation="relu"),
            BatchNormalization(),
        ]
        self.t1_max_pool = MaxPool1D(pool_size=10)
        self.t1_mlp2 = [
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Flatten(),
            Dense(9)
        ]

        # Feature transformation subnet
        self.t2_mlp = [
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dense(1024, activation="relu"),
            BatchNormalization(),
        ]
        self.t2_max_pool = MaxPool1D(pool_size=10)
        self.t2_mlp2 = [
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Flatten(),
            Dense(64*64)
        ]

        self.mlp1 = Dense(64)

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

    # @tf.function
    def call(self, x, training=False):
        x = self.ae(x)

        # x = self.input_transform(x)
        x = self.mlp1(x)
        x = self.feature_transform(x)

        for layer in self.arch:
            x = layer(x, training=training)

        return x

    @tf.function
    def input_transform(self, x):
        x1, y1, z1 = x[:, :, 0], x[:, :, 1], x[:, :, 2]

        for layer in self.t1_mlp:
            x1 = layer(x1)
            y1 = layer(y1)
            z1 = layer(z1)

        t = tf.stack((x1, y1, z1), axis=-1)
        t = self.t1_max_pool(t)

        for layer in self.t1_mlp2:
            t = layer(t)

        t = tf.reshape(t, (-1, 3, 3))
        x = tf.matmul(x, t)
        return x

    @tf.function
    def feature_transform(self, x):
        x1, y1, z1 = x[:, :, 0], x[:, :, 1], x[:, :, 2]

        for layer in self.t2_mlp:
            x1 = layer(x1)
            y1 = layer(y1)
            z1 = layer(z1)

        t = tf.stack((x1, y1, z1), axis=-1)
        t = self.t2_max_pool(t)

        for layer in self.t2_mlp2:
            t = layer(t)

        t = tf.reshape(t, (-1, 64, 64))
        x = tf.matmul(x, t)
        return x
