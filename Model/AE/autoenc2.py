import numpy as np
import h5py
from keras import utils
from keras import layers, Model, Input

# =========================================================
# Generator
# =========================================================

class TrajectoryGenerator(utils.Sequence):
    def __init__(self, h5_path, batch_size=64, shuffle=True):
        self.h5 = h5py.File(h5_path, "r")
        self.X = self.h5["X"]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))

        if shuffle:
            np.random.shuffle(self.indices)

        self.T = self.X.shape[1]  # 360
        self.F = self.X.shape[2]  # 8

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]

        batch = np.empty((len(batch_idx), self.T, self.F), dtype=np.float32)
        for i, bi in enumerate(batch_idx):
            batch[i] = self.X[bi]

        return batch, batch   # AE target = input

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# Autoencoder Architecture (Optimized for Clustering)
# =========================================================

T = 360
F = 8
latent_dim = 64

inp = Input(shape=(T, F))

# -------- Encoder (NO batchnorm, better for clustering) --------
x = layers.Conv1D(32, 5, padding='same', activation='relu')(inp)       # 360
x = layers.Conv1D(64, 5, strides=2, padding='same', activation='relu')(x) # 360 → 180
x = layers.Conv1D(128, 5, strides=2, padding='same', activation='relu')(x) # 180 → 90
x = layers.Conv1D(256, 5, strides=2, padding='same', activation='relu')(x) # 90 → 45

x = layers.Flatten()(x)
latent = layers.Dense(latent_dim, activation=None, name="latent")(x)


# -------- Decoder --------
x = layers.Dense(45 * 256)(latent)
x = layers.Reshape((45, 256))(x)

x = layers.UpSampling1D(2)(x)      # 45 → 90
x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)

x = layers.UpSampling1D(2)(x)      # 90 → 180
x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)

x = layers.UpSampling1D(2)(x)      # 180 → 360
x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)

out = layers.Conv1D(F, 3, padding='same', activation='linear')(x)

autoencoder = Model(inp, out)
encoder = Model(inp, latent)

autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.summary()


# =========================================================
# Train
# =========================================================

gen = TrajectoryGenerator("trajectoriesNorm.h5", batch_size=64)

autoencoder.fit(
    gen,
    epochs=25,
    verbose=1
)

# =========================================================
# Encode all trajectories efficiently
# =========================================================

with h5py.File("trajectoriesNorm.h5", "r") as h5:
    X = h5["X"]
    # Vectorized batch encoding
    latent_vectors = encoder.predict(X, batch_size=512, verbose=1)

np.save("latent_vectors.npy", latent_vectors)
