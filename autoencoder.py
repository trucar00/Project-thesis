import numpy as np
import h5py
from keras import utils
from keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf

class TrajectoryGenerator(utils.Sequence):
    def __init__(self, h5_path, indices, batch_size=64, shuffle=True):
        self.h5 = h5py.File(h5_path, "r")
        self.X = self.h5["X"]

        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.T = self.X.shape[1]
        self.F = self.X.shape[2]

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
    
        batch = np.empty((len(batch_idx), self.T, self.F), dtype=np.float32)
        for i, bi in enumerate(batch_idx):
            batch[i] = self.X[bi]
    
        return batch, batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# -- 3h trajectories --

def architecture_3h(T=360, F=8, latent_dim=64): # 3h * 60 * 2 = 260 timesteps, F = 8 features

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

    return autoencoder, encoder

# -- 6h trajectories --

def architecture_6h(T=720, F=8, latent_dim=64): # 6h * 60 * 2 = 720, F = 8 features

    inp = Input(shape=(T, F))

    # -------- Encoder --------
    x = layers.Conv1D(32, 5, padding='same', activation='relu')(inp)            # 720
    x = layers.Conv1D(64, 5, strides=2, padding='same', activation='relu')(x)   # 720 → 360
    x = layers.Conv1D(128, 5, strides=2, padding='same', activation='relu')(x)  # 360 → 180
    x = layers.Conv1D(256, 5, strides=2, padding='same', activation='relu')(x)  # 180 → 90

    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation=None, name="latent")(x)

    # -------- Decoder --------
    x = layers.Dense(90 * 256)(latent)          # match last encoder length (90)
    x = layers.Reshape((90, 256))(x)

    x = layers.UpSampling1D(2)(x)               # 90 → 180
    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)

    x = layers.UpSampling1D(2)(x)               # 180 → 360
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)

    x = layers.UpSampling1D(2)(x)               # 360 → 720
    x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)

    out = layers.Conv1D(F, 3, padding='same', activation='linear')(x)

    autoencoder = Model(inp, out)
    encoder = Model(inp, latent)

    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.summary()

    return autoencoder, encoder


def train_model(trajectories_path, autoencoder, encoder, ae_path):
    print("Training model.")
    with h5py.File(trajectories_path, "r") as h5:
        N = len(h5["X"])

    print("Nr of trajectories in h5 file for training: ", N)

    train_idx, val_idx = train_test_split(
        np.arange(N), 
        test_size=0.1,
        shuffle=True,
        random_state=42
    )

    gen_train = TrajectoryGenerator(trajectories_path, train_idx, batch_size=64)
    gen_val   = TrajectoryGenerator(trajectories_path, val_idx, batch_size=64, shuffle=False)

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = autoencoder.fit(
        gen_train,
        validation_data=gen_val,
        epochs=200,
        callbacks=[es],
        verbose=1
    )

    autoencoder.save(ae_path)

    with h5py.File(trajectories_path, "r") as h5:
        X_all = h5["X"]
        # Vectorized batch encoding
        latent_vectors = encoder.predict(X_all, batch_size=512, verbose=1)

    np.save(ae_path, latent_vectors)
    print("Successfully saved the latent vectors")


def main(trajectories_path, ae_path, traj_length):
    print("Setting up architecture.")
    if traj_length == 3:
        autoencoder, encoder = architecture_3h()
    elif traj_length == 6:
        autoencoder, encoder = architecture_6h()
    else:
        print("Please specify either the 3h or 6h trajectories.")
        return
    
    train_model(trajectories_path, autoencoder, encoder, ae_path)


if __name__ == "__main__":
    print("Please specify trajectory path for training, ae_path for saving the autoencoder and latent path for saving the latent vectors.")