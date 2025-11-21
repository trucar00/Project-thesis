import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Processed_AIS/Resampled2/2024-03.csv")
df2 = pd.read_csv("Processed_AIS/TrajLonger2h/2024-03.csv")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df2["date_time_utc"] = pd.to_datetime(df2["date_time_utc"])
df.sort_values(by="date_time_utc")
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

