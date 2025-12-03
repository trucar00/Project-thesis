import pandas as pd
import h5py

df = pd.read_csv("Featureset/2024Feats.csv")

groups = df.groupby("trajectory_id")
N = len(groups)
T = 3*60*2
F = 6

h5 = h5py.File("trajectories.h5", "w")
dset = h5.create_dataset("X", shape=(N, T, F), dtype="float32")

index = 0
for tid, g in groups:
    g = g.sort_values("date_time_utc")
    arr = g[["speed","del_cog","rot","accel_bwd","avg_speed","std_speed"]].values
    dset[index] = arr
    index += 1

h5.close()