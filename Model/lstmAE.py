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

# try with one sequence

df = pd.read_csv("../Featureset/2024FeatsNorm.csv", nrows=360) # first trajectory

train = df[["avg_speed", "std_speed", "z_speed", "z_del_cog", "z_rot", "z_accel_bwd", "lon_rel", "lat_rel"]].values
# samples = 1 (1 trajectory)
# timesteps = 360
# features = 8

n_in = train.shape[0]
n_features = train.shape[1]

train = train.reshape(1, n_in, n_features)

# Encoder
visible = Input(shape=(n_in, n_features))
encoder = LSTM(100, activation="relu")(visible)

# Decoder
decoder = RepeatVector(n_in)(encoder)
decoder = LSTM(100, activation="relu", return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(n_features))(decoder)

model = Model(inputs=visible, outputs=decoder)
model.compile(optimizer="adam", loss="mse")
print(model.summary())


model.fit(train, train, epochs=300, verbose=1)
yhat = model.predict(train)

# Plot reconstructed yhat and real