import pandas as pd
from time import time
import os
import cleanAIS
from sklearn.preprocessing import StandardScaler


def angle_wrap(a):
    
    return (a + 180) % 360 - 180


def build(df):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df["del_cog"] = df.groupby("trajectory_id")["cog"].diff().apply(angle_wrap)
    df["del_cog"] = df["del_cog"].fillna(0)
    df["accel_bwd"] = df["accel_bwd"].fillna(0)
    df["rot"] = df["rot"].fillna(0)

    window_length = pd.Timedelta(hours=6)
    step = pd.Timedelta(minutes=2*60) # Good value ?
    
    all_windows = []
    

    for traj_id, d in df.groupby("trajectory_id"):
        d = d.sort_values("date_time_utc").reset_index(drop=True)

        start = d["date_time_utc"].min()
        end = d["date_time_utc"].max()
        current = start
        window_id = 0

        while current + window_length <= end:
            window_df = d[(d["date_time_utc"] >= current)
                         & (d["date_time_utc"] < (current + window_length))].copy()
            
            window_df["trajectory_id"] = window_df["trajectory_id"].astype(str) + "-" + str(window_id)
            window_df["lon_rel"] = window_df["lon"] - window_df["lon"].iloc[0]
            window_df["lat_rel"] = window_df["lat"] - window_df["lat"].iloc[0]
            

            feature_df = window_df[["trajectory_id", "mmsi", "date_time_utc", "speed", "del_cog", "rot", "accel_bwd", "lon_rel", "lat_rel"]].copy()
            feature_df["window_start"] = current
            feature_df["window_end"] = current + window_length
            feature_df["avg_speed"] = window_df["speed"].mean() # Could remove avg and std speed thus reducing nr of features to 6
            feature_df["std_speed"] = window_df["speed"].std()

            all_windows.append(feature_df)

            current += step
            window_id += 1

    df_all = pd.concat(all_windows, ignore_index=True)
    df_all = cleanAIS.reindex_trajectory_ids(df_all)
    df_all = df_all.sort_values(by=["trajectory_id", "window_start"])

    df_all = df_all[["trajectory_id", "mmsi", "window_start", "window_end", "avg_speed", "std_speed",
                     "date_time_utc", "speed", "del_cog", "rot", "accel_bwd", "lon_rel", "lat_rel"]] # # add avg std
    return df_all

def createCommon():
    dfs = []
    for month in range(1,13):
        filepath = f"Featureset/6h_sin_avg_std/2024-{month:02d}.csv"
        df = pd.read_csv(filepath)
        df["trajectory_id"] = (df["trajectory_id"].astype(str) + "-" + f"{month}")
        print(df["trajectory_id"].iloc[0])
        dfs.append(df)

    concat_df = pd.concat(dfs, ignore_index=True)
    scaler = StandardScaler()
    scaled_cols = ["speed", "del_cog", "rot", "accel_bwd", "lon_rel", "lat_rel"] # add avg std
    concat_df[scaled_cols] = scaler.fit_transform(concat_df[scaled_cols])
    concat_df.rename(columns={c: f"z_{c}" for c in scaled_cols}, inplace=True)
    
    concat_df.to_csv("Featureset/2024Feats6hsin.csv", index=False)

    return "Done!"


def main():
    start = time()

    for month in range(1,13):
        getfile = f"Processed_AIS/Resampled6h/2024-{month:02d}.csv"
        savefile = f"Featureset/6h_sin_avg_std/2024-{month:02d}.csv"

        if os.path.exists(getfile):
            print("Building features for: ", getfile)
            df = pd.read_csv(getfile, engine="pyarrow")
            df_feats = build(df)
            df_feats.to_csv(savefile, index=False)
            print("Saved features to: ", savefile)          
        else:
            print("Missing: ", getfile)

    createCommon()
    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")
    return

if __name__ == "__main__":
    main()
    