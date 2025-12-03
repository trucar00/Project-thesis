import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# -- Plot example trajectories of the 1000 ish --
with open("traj_clusters.json", "r") as f:
    cluster_dict = json.load(f)

# This is the island cluster (~1000 trajectories)
cluster_id = "1"   # cluster_dict keys are strings
trajectory_ids = cluster_dict[cluster_id]
print("Trajectories in cluster:", len(trajectory_ids))

df = pd.read_csv("../Featureset/2024FeatsNorm.csv")

# Filter only columns we need
cols = ["trajectory_id", "lon_rel", "lat_rel", "date_time_utc", "avg_speed"]
df_small = df[cols]

# -------------------------
# Pick N example trajectories
# -------------------------
N = 9  # plot 9 example trajectories
example_ids = np.random.choice(trajectory_ids, size=N, replace=False)

# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for ax, traj_id in zip(axes, example_ids):
    print(traj_id)
    # select the rows for this trajectory, sorted by timestep
    d = df_small[df_small["trajectory_id"] == traj_id].copy()
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values("date_time_utc")
    print("Avg speed: ", d["avg_speed"].iloc[0])

    ax.plot(d["lon_rel"], d["lat_rel"], marker='.', markersize=1)
    ax.set_title(f"Traj {traj_id}", fontsize=8)
    ax.axis("equal")
    ax.axis("off")   # cleaner look

# Hide any unused axes
for ax in axes[len(example_ids):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

