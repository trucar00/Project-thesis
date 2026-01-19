import pandas as pd
import h5py

# Can be a part of buildTrainingSet!

def createh5(training_path, traj_length, trajectories_path):

    print("Creating h5 file.")
    df = pd.read_csv(f"{training_path}2024.csv") # or 2024FullTrainingSet

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    groups = df.groupby("trajectory_id")
    N = len(groups)
    T = traj_length*60*2
    F = 8

    h5 = h5py.File(trajectories_path, "w")
    dset = h5.create_dataset("X", shape=(N, T, F), dtype="float32")

    index = 0

    for tid, g in groups:

        g = g.sort_values("date_time_utc")
        arr = g[["z_speed","z_del_cog","z_rot","z_accel_bwd", "z_avg_speed", "z_std_speed", "z_lon_rel", "z_lat_rel"]].values
        dset[index] = arr
        index += 1

    h5.close()
    print("Done!")