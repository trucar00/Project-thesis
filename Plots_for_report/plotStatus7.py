import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

""" Found that the status flag 7 is not reliable. It is clear that fishing vessels engaged in fishing activity forget to set this flag. 
And also forget to turn it off when done with fishing. This plot shows status flag 7 vessels in red and all fishing vessels (ship_type=30) trajectories (after cleaning) in blue."""

df = pd.read_csv("./Processed_AIS/Resampled2/2024-03.csv")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df.sort_values(by="date_time_utc")

#df2 = pd.read_csv("Processed_AIS/TrajLonger2h/2024-03.csv")
df2 = df.loc[df["status"] == 7].copy()
df2["date_time_utc"] = pd.to_datetime(df2["date_time_utc"])
df2.sort_values(by="date_time_utc")

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
used_labels = set()

for traj_id, traj in df.groupby("trajectory_id"):

    ax.plot(traj["lon"], traj["lat"], linewidth=1, color="blue")

for traj_id, traj in df2.groupby("trajectory_id"):

    ax.plot(traj["lon"], traj["lat"], linewidth=1, color="red")

ax.legend()
plt.show()

