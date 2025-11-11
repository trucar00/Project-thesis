## Code for downsampling AIS data

import pandas as pd

def resample(df, step="30s"):
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
df_resampled = resample(df, step="30s")
print(df_resampled.head())
print(df_resampled.tail())

# Something off with the mmsi. Want to downsample for every mmsi. Problem is that if step is 30s and we start at first instance for mmsi x it will have a fixed stepsize 30s.
# So for the next 30s if an AIS-message dont align with the step it gives NaN values. Some other way to resample?