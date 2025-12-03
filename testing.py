import pandas as pd
import matplotlib.pyplot as plt
import cleanAIS

""" df = pd.read_parquet("Processed_AIS/Concatenated/2024-01.parquet", engine="pyarrow")

df2 = df[df["mmsi"] == 259536000]

df2["date_time_utc"] = pd.to_datetime(df2["date_time_utc"])

df2.sort_values(by="date_time_utc")

df2.to_csv("weird.csv") """

""" df = pd.read_csv("weird.csv")

plt.scatter(df["lon"], df["lat"], s=1)
plt.show()

df = cleanAIS.all(df)

plt.scatter(df["lon"], df["lat"], s=1)
plt.show() """

df = pd.read_csv("Processed_AIS/Resampled2/2024-01.csv")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df = df.sort_values(by=["trajectory_id", "date_time_utc"])

print("NR:", df["trajectory_id"].nunique())

i = 0

for traj_id, d in df.groupby("trajectory_id"):
    i+=1
    mini = d["date_time_utc"].min()
    maxi = d["date_time_utc"].max()
    time = maxi - mini
    print("Trajectory ID: ", traj_id, " Start: ", mini, " End: ", maxi, " Delta: ", time)

print(i)