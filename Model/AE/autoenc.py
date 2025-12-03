import numpy as np
import h5py
from keras import utils
from keras import layers, Model, Input

class TrajectoryGenerator(utils.Sequence):
    def __init__(self, h5_path, batch_size=64, shuffle=True):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, "r")
        self.X = self.h5["X"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if shuffle:
            np.random.shuffle(self.indices)

        self.T = self.X.shape[1]
        self.F = self.X.shape[2]

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        # allocate batch manually
        batch = np.empty((len(batch_idx), self.T, self.F), dtype=np.float32)

        # fill batch one trajectory at a time
        for i, bi in enumerate(batch_idx):
            batch[i] = self.X[bi]

        return batch, batch  # AE target = input

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

gen = TrajectoryGenerator("trajectories.h5", batch_size=64) 

T = 360
F = 8
latent_dim = 32

inp = Input(shape=(T, F))

# ----- Encoder -----
x = layers.Conv1D(filters=32, kernel_size=5, strides=1, activation='relu', padding='same')(inp)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(filters=128, kernel_size=9, strides=3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
latent = layers.Dense(latent_dim, activation=None)(x)

# ----- Decoder -----
x = layers.Dense(60 * 128)(latent)
x = layers.Reshape((60, 128))(x)
x = layers.UpSampling1D(3)(x)     # 60 -> 180
x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
x = layers.UpSampling1D(2)(x)     # 180 -> 360
x = layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
x = layers.Conv1D(filters=32, kernel_size=9, activation='relu', padding='same')(x)

out = layers.Conv1D(filters=F, kernel_size=3, activation='linear', padding='same')(x)

autoencoder = Model(inp, out)
encoder = Model(inp, latent)

autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.summary()

autoencoder.fit(
    gen,
    epochs=25,
    verbose=1
)

h5 = h5py.File("trajectories.h5", "r")
X = h5["X"]

latent_vectors = []

for i in range(len(X)):
    traj = X[i:i+1]         # keep batch dimension
    z = encoder.predict(traj, verbose=0)
    latent_vectors.append(z[0])

latent_vectors = np.array(latent_vectors)
np.save("latent_vectors.npy", latent_vectors)