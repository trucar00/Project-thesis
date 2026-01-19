import pandas as pd
from time import time
import os

# --- Code for downsampling AIS data ---

def downsample(df, step):
    # Ensure datetime format
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"])
    df = df.set_index("date_time_utc")

    def resample_and_interpolate(g):
        # Resample regularly
        g_res = g.resample(step, origin=g.index.min()).first()

        # Interpolate only numeric columns (lon, lat, speed, etc.)
        num_cols = g_res.select_dtypes(include="number").columns
        g_res[num_cols] = g_res[num_cols].interpolate(method="linear") # Linear interpolation to fill missing values

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


def main(cleaned_path, resmapled_path, step):
    start = time()

    for month in range(1,13):
        getfile = f"{cleaned_path}{month:02d}.csv"
        savefile = f"{resmapled_path}{month:02d}.csv"
        if os.path.exists(getfile):
            print("Resampling: ", getfile)
            df = pd.read_csv(getfile, engine="pyarrow")
            df = downsample(df, step)
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