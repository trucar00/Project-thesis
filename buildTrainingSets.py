import pandas as pd
from time import time
import os

# PLAN
# Build features for all trajectories. Use sliding window.
# One big dataset?
# How does the unsupervised work? I want to cluster all "similar" trajectories

# Features
# speed, mean speed, std speed, acceleration
# deltaCOG, dCOG/dt, ROT

def angle_wrap(a): # Understand this
    
    return (a + 180) % 360 - 180

def build(df):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df["del_cog"] = df.groupby("trajectory_id")["cog"].diff().apply(angle_wrap)
    df["del_cog"] = df["del_cog"].fillna(0)
    df["accel_bwd"] = df["accel_bwd"].fillna(0)

    window_length = pd.Timedelta(hours=2)
    step = pd.Timedelta(minutes=30) # Good value?
    
    all_windows = []

    for traj_id, d in df.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc").reset_index(drop=True)

        start = d["date_time_utc"].min()
        end = d["date_time_utc"].max()
        current = start

        while current + window_length <= end:
            window_df = d[(d["date_time_utc"] >= current)
                         & (d["date_time_utc"] < (current + window_length))].copy()
            
            feature_df = window_df[["trajectory_id", "mmsi", "date_time_utc", "speed", "del_cog", "rot", "accel_bwd"]].copy()
            feature_df["window_start"] = current
            feature_df["window_end"] = current + window_length
            feature_df["avg_speed"] = window_df["speed"].mean()
            feature_df["std_speed"] = window_df["speed"].std()

            all_windows.append(feature_df)

            current += step

    df_all = pd.concat(all_windows, ignore_index=True)
    df_all = df_all[["trajectory_id", "mmsi", "window_start", "window_end", "avg_speed", "std_speed", "date_time_utc", "speed", "del_cog", "rot", "accel_bwd"]] # Add acceleration?
    return df_all


def main():
    start = time()

    for month in range(1,13):
        getfile = f"Processed_AIS/Resampled2/2024-{month:02d}.csv"
        savefile = f"Featureset/2024-{month:02d}.csv"

        if os.path.exists(getfile):
            print("Building features for: ", getfile)
            df = pd.read_csv(getfile, engine="pyarrow")
            df_feats = build(df)
            df_feats.to_csv(savefile, index=False)
            print("Saved features to: ", savefile)          
        else:
            print("Missing: ", getfile)

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")
    return

if __name__ == "__main__":
    main()