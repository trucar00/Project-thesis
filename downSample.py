## Code for downsampling AIS data

import pandas as pd
from time import time
import os

""" def resample(df, step="30s"):
    print("Resampling")
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    #"df["mmsi"] = df["mmsi"].astype("int64")
    check_nan = df["mmsi"].isna().sum()
    print(check_nan, " NaN values in mmsi column")
    df = df.sort_values(by=["mmsi", "date_time_utc"])
    df = df.set_index("date_time_utc")
    #print(df.head())

    resampled = (
        df.groupby("mmsi", group_keys=False)
          .apply(lambda g: g.resample(step, origin=g.index.min()).first())
    )

    try:

        resampled["mmsi"] = resampled["mmsi"].astype("int64")
    except Exception as e:
        print("Could not convert mmsi to int64: ", e)

    resampled = resampled.reset_index()

    return resampled


df = pd.read_csv("Processed_AIS/Cleaned/2024-01.csv")
#df_resampled = resample(df, step="30s")


def downsample_ais(df, step="30s"):
    # Ensure datetime format
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"])
    df = df.set_index("date_time_utc")

    # Resample each MMSI separately
    resampled = (
        df.groupby("trajectory_id", group_keys=False)
          .apply(lambda g: g.resample(step, origin=g.index.min()).first())
    )

    # Fill in MMSI (lost due to NaN in resampling)
    resampled["mmsi"] = resampled["mmsi"].fillna(method="ffill").astype("int64")
    resampled = resampled.reset_index()

    return resampled """

def downsample(df, step="30s"):
    # Ensure datetime format
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"])
    df = df.set_index("date_time_utc")

    def resample_and_interpolate(g):
        # Resample regularly
        g_res = g.resample(step, origin=g.index.min()).first()

        # Interpolate only numeric columns (lon, lat, speed, etc.)
        num_cols = g_res.select_dtypes(include="number").columns
        g_res[num_cols] = g_res[num_cols].interpolate(method="linear")

        # Fill remaining NaNs (like mmsi, ship_name) via forward/backward fill
        g_res = g_res.ffill().bfill()
        return g_res

    # Apply per trajectory
    resampled = df.groupby("trajectory_id", group_keys=False).apply(resample_and_interpolate)

    # Ensure mmsi is valid integer
    resampled["mmsi"] = resampled["mmsi"].astype("int64")

    # Reset index for output
    resampled = resampled.reset_index()

    return resampled


def main():
    start = time()

    for month in range(1,13):
        getfile = f"Processed_AIS/Cleaned/2024-{month:02d}.csv"
        savefile = f"Processed_AIS/Resampled/2024-{month:02d}.csv"
        if os.path.exists(getfile):
            print("Resampling: ", getfile)
            df = pd.read_csv(getfile, engine="pyarrow")
            df = downsample(df)
            df.to_csv(savefile, index=False)
            print(f"Saved resampled data for 2024-{month:02d} to {savefile}")          
        else:
            print("Missing: ", getfile)

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")
    return

if __name__ == "__main__":
    #main()
    print("Already created resampled csv's")