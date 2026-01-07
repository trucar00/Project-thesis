import pandas as pd
import h5py

df = pd.read_csv("../Featureset/2024FeatsNorm.csv")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

groups = df.groupby("trajectory_id")
N = len(groups)
T = 360
F = 8

h5 = h5py.File("Data/trajectoriesNorm.h5", "w")
dset = h5.create_dataset("X", shape=(N, T, F), dtype="float32")

index = 0

for tid, g in groups:

    g = g.sort_values("date_time_utc")
    arr = g[["z_speed","z_del_cog","z_rot","z_accel_bwd","avg_speed","std_speed", "lon_rel", "lat_rel"]].values
    dset[index] = arr
    index += 1

h5.close()