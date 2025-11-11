import os
import dataProcessing
from time import time
import pandas as pd
import numpy as np
import math


def remove_stationary(df, speed_threshold=0.4, min_duration="10min"):
    print("Removing stationary")

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(by=["mmsi", "date_time_utc"])

    df["stationary"] = df["speed"] < speed_threshold
    df["grp"] = (df["stationary"] != df["stationary"].shift()).cumsum()

    drop = []
    for _, g in df[df["stationary"]].groupby("grp"):

        duration = g.index.max() - g.index.min()
        if duration > pd.Timedelta(min_duration).seconds:
            drop.append(g.index)

    df = df.drop(pd.Index(np.concatenate(drop)))
    return df.drop(columns=["stationary", "grp"])

def remove_invalid(df, min_cog=0, max_cog=360, min_speed=0, max_speed=30):
    print("Removing invalid rows")

    # Ensure numeric columns
    for col in ["cog", "speed", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a mask for valid values
    valid_mask = (
        df["cog"].between(min_cog, max_cog, inclusive="both")
        & df["speed"].between(min_speed, max_speed, inclusive="both")
    )

    invalid_count = len(df) - valid_mask.sum()
    print(f"Removed {invalid_count:,} invalid rows")

    return df[valid_mask]

def extract_trajectories(df, time_threshold="30min"):
    df = df.sort_values(["mmsi", "date_time_utc"])
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df["dt"] = df.groupby("mmsi")["date_time_utc"].diff().dt.total_seconds()
    tt = pd.Timedelta(time_threshold).total_seconds()

    df["traj_id"] = (df["dt"] > tt).groupby(df["mmsi"]).cumsum()
    df["trajectory_id"] = df["mmsi"].astype(str) + "-" + df["traj_id"].astype(str)

    return df.drop(columns=["dt"])
    

def resample(df, sample="1min"): # not finished
    dfs = []
    for _, g in df.groupby("mmsi"):

        g = g.set_index("date_time_utc")
        g = g.resample(sample, origin="start").first()
        print(g)
    return dfs

def remove_sparse_trajectories(df, interval=30):
    print(f"Removing trajectories with message interval > {interval} seconds")

    df = df.sort_values(["trajectory_id", "date_time_utc"])
    df["dt"] = df.groupby("trajectory_id")["date_time_utc"].diff().dt.total_seconds()
    too_sparse = df.groupby("trajectory_id")["dt"].max() > interval
    valid_traj = too_sparse[~too_sparse].index
    df_filtered = df[df["trajectory_id"].isin(valid_traj)]

    return df_filtered.drop(columns=["dt"])

def remove_trajectories_few_instances(df, min_instances=100):
    print(f"Removing trajectories with fewer than {min_instances} messages")

    counts = df["trajectory_id"].value_counts()
    valid_traj = counts[counts >= min_instances].index
    df_filtered = df[df["trajectory_id"].isin(valid_traj)]

    removed = len(counts) - len(valid_traj)
    print(f"Removed {removed} trajectories")

    return df_filtered

def remove_duplicate_timestamps(df):
    print("Removing duplicate timestamps per trajectory")

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    before = len(df)

    # keep only the row with highest speed for each (trajectory_id, timestamp)
    df = (
        df.sort_values(["trajectory_id", "date_time_utc", "speed"], ascending=[True, True, False])
          .drop_duplicates(subset=["trajectory_id", "date_time_utc"], keep="first")
    )

    removed = before - len(df)
    print(f"Removed {removed:,} duplicate-timestamp rows")
    return df

def haversine(lat1, lon1, lat2, lon2, dt):
    R = 6371000 # Radius of the earth in meters

    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) + 
         pow(np.sin(dLon / 2), 2) * 
             np.cos(lat1) * np.cos(lat2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    speed = (dist/dt) * 1.94384 # Convert m/s to knots

    return dist, speed

def remove_outlier_positions(df, max_speed = 30):
    print("Removing trajectory outliers (impossible jumps)")
    df = df.sort_values(["trajectory_id", "date_time_utc"])
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df["dt_fwd"] = df.groupby("trajectory_id")["date_time_utc"].diff().shift(-1).dt.total_seconds()
    df["dt_bwd"] = df.groupby("trajectory_id")["date_time_utc"].diff().dt.total_seconds()

    df["lat_prev"] = df.groupby("trajectory_id")["lat"].shift()
    df["lon_prev"] = df.groupby("trajectory_id")["lon"].shift()
    df["lat_next"] = df.groupby("trajectory_id")["lat"].shift(-1)
    df["lon_next"] = df.groupby("trajectory_id")["lon"].shift(-1)

    df["del_m_fwd"], df["speed_fwd"] = haversine(df["lat"], df["lon"], df["lat_next"], df["lon_next"], df["dt_fwd"])
    df["del_m_bwd"], df["speed_bwd"] = haversine(df["lat_prev"], df["lon_prev"], df["lat"], df["lon"], df["dt_bwd"])

    df["del_speed_fwd"] = df.groupby("trajectory_id")["speed"].diff().shift(-1)
    df["del_speed_bwd"] = df.groupby("trajectory_id")["speed"].diff()
    
    df["accel_fwd"] = (df["del_speed_fwd"] / 1.94384) / df["dt_fwd"]
    df["accel_bwd"] = (df["del_speed_bwd"] / 1.94384) / df["dt_bwd"]


    jump_mask = (df["speed_bwd"] > max_speed) & (df["speed_fwd"] > max_speed)
    accel_mask = (df["accel_fwd"].abs() > 0.15) & (df["accel_bwd"].abs() > 0.15)

    outlier_mask = jump_mask | accel_mask

    df_filtered = df[~outlier_mask].drop(columns=[
        "dt_fwd", "dt_bwd",
        "lat_prev", "lon_prev", "lat_next", "lon_next",
        "del_m_fwd", "del_m_bwd", "del_speed_fwd", "del_speed_bwd", "speed_fwd", "speed_bwd"
    ])

    print(f"Removed {outlier_mask.sum():,} outlier points")

    return df_filtered

def reindex_trajectory_ids(df):
    print("Reindexing trajectory IDs")

    df = df.sort_values(["mmsi", "date_time_utc"])

    # Map old trajectory IDs to new sequential ones per MMSI
    new_ids = []
    for mmsi, group in df.groupby("mmsi"):
        unique_trajs = {old_id: new_id for new_id, old_id in enumerate(sorted(group["trajectory_id"].unique()))}
        new_ids.append(group.assign(
            trajectory_id_new=group["trajectory_id"].map(unique_trajs),
            trajectory_id=lambda g: g["mmsi"].astype(str) + "-" + g["trajectory_id_new"].astype(str)
        ))

    df = pd.concat(new_ids, ignore_index=True)
    df = df.drop(columns=["trajectory_id_new"])
    return df

def all(df):
    df = remove_invalid(df)
    df = remove_stationary(df)
    df = extract_trajectories(df)
    df = remove_sparse_trajectories(df)
    df = remove_trajectories_few_instances(df)
    df = reindex_trajectory_ids(df)
    df = remove_duplicate_timestamps(df)
    df = remove_outlier_positions(df)
    return df

def main():
    df = pd.read_parquet("Processed_AIS/Concatenated/2024-02.parquet", engine="pyarrow")
    print(df.shape)
    df = remove_invalid(df)
    print(df.shape)
    df = remove_stationary(df)
    print(df.shape)
    df = extract_trajectories(df)
    print(df.shape)
    
    df = remove_sparse_trajectories(df)
    print(df.shape)
    df = remove_trajectories_few_instances(df)
    print(df.shape)
    df = reindex_trajectory_ids(df)
    df = remove_duplicate_timestamps(df)
    df = remove_outlier_positions(df)
    df.to_csv("Processed_AIS/Resampled/2024-02.csv", index=False)

    return

if __name__ == "__main__":
    main()
    #print(haversine(69.65481, 18.970306, 69.60507, 18.88542, 1))
    #df = pd.read_csv("Processed_AIS/Resampled/2024-01_02.csv")
    #df = remove_outlier_positions(df)
    #df.to_csv("test2.csv")
    
# TODO
# Remove ships with too few instances after each other. There should be some consecutive instances over some period of time. Maybe filtering out ships who sends out ais too rarely
# Remove outlier points (noise)
# Linear interpolation 
# Autoencoder
# Remove rows with the same timestamp!!
