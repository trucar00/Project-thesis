import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.utils import model_to_dot
import pandas as pd
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# try with one sequence
n_samples = 2
rows = 360 * n_samples
df = pd.read_csv("../Featureset/2024FeatsNorm.csv", nrows=rows) # first trajectory

traj_ids = df["trajectory_id"].unique()
#traj_ids = df["trajectory_id"].unique()[:n_samples] # trying with first 10
print(len(traj_ids)) # 54658 for full dataset

feature_cols = ["avg_speed", "std_speed", "z_speed", "z_del_cog", "z_rot", "z_accel_bwd", "lon_rel", "lat_rel"]

sequences = []

for tid in traj_ids:
    traj = df[df["trajectory_id"] == tid][feature_cols].values
    #print(traj.shape) # (360, 8)
    sequences.append(traj)

train = np.stack(sequences, axis=0)

# samples = 10 (10 trajectories)
# timesteps = 360
# features = 8

print(train.shape) # (20, 360, 8) = (samples, timesteps, features)

n_in = train.shape[1] 
n_features = train.shape[2]



# Encoder
visible = Input(shape=(n_in, n_features))
encoder = LSTM(32, activation="relu")(visible)

# Decoder
decoder = RepeatVector(n_in)(encoder)
decoder = LSTM(32, activation="relu", return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(n_features))(decoder)

optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
model = Model(inputs=visible, outputs=decoder)
model.compile(optimizer=optimizer, loss="mse")
print(model.summary())


model.fit(train, train, epochs=100, verbose=1, shuffle=True) # batch_size = 32  default
yhat = model.predict(train)

#print(yhat)

# Plot reconstructed yhat and real

feature_names = ["avg_speed", "std_speed", "z_speed", "z_del_cog",
                 "z_rot", "z_accel_bwd", "lon_rel", "lat_rel"]

n_traj = train.shape[0]          # number of trajectories (e.g., 20)
n_features = train.shape[2]      # = 8
timesteps = train.shape[1]       # = 360

for idx in range(n_traj):
    train_seq = train[idx]       # shape (360, 8)
    yhat_seq = yhat[idx]         # shape (360, 8)
    print("Traj 0 mean:", np.mean(train_seq[0]))
    print("Traj 1 mean:", np.mean(train_seq[1]))
    print("Traj 2 mean:", np.mean(train_seq[2]))

    print("Recons 0 mean:", np.mean(yhat_seq[0]))
    print("Recons 1 mean:", np.mean(yhat_seq[1]))
    print("Recons 2 mean:", np.mean(yhat_seq[2]))

    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Trajectory ID: {traj_ids[idx]}", fontsize=16)

    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        plt.plot(train_seq[:, i], label="Original", color='blue', linewidth=1)
        plt.plot(yhat_seq[:, i], label="Reconstructed", color='red', linestyle='--', linewidth=1)
        plt.title(feature_names[i])
        plt.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for main title
    plt.show()