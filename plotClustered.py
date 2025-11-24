import pandas as pd
import matplotlib.pyplot as plt
"""-1: 50894, 0: 1148, 1: 448, 2:362, 3: 1486, 4: 103, 5: 171, 6: 46"""
df = pd.read_csv("Featureset/2024FeatsPos.csv")

print(df['trajectory_id'].nunique())

# Cluster 3
c3 = ['257073100-11-1', '257073100-12-1', '257073100-13-1', '257073100-14-1', '257073100-15-1']

# Cluster 5
c5 = ['257580600-31-1', '257064260-6-3', '273370020-10-3', '273370020-32-3', '273370020-33-3']

# Cluster 0
c0 = ['259617000-10-1', '259617000-12-1', '259617000-13-1', '259617000-14-1', '259617000-15-1']

# Cluster 2
c2 = ['259617000-5-1', '257271600-50-2', '257271600-78-2', '257271600-88-2', '257305000-11-2']

# Cluster 1
c1 = ['259617000-8-1', '257271600-2-2', '257271600-52-2', '257271600-62-2', '257271600-72-2']

# Cluster 6
c6 = ['259653000-4-1', '259655000-2-1', '259655000-6-1', '259660000-1-1', '259660000-12-1']

# Cluster 4
c4 = ['273448380-92-2', '273890200-33-2', '251201000-24-3', '257413500-2-3', '273418680-19-3']


fig, axes = plt.subplots(1, 5, figsize=(10, 3))
axes = axes.flatten()  # make it easy to index

for ax, traj_id in zip(axes, c6):

    traj_data = df[df["trajectory_id"] == traj_id]
    avg_speed = traj_data["avg_speed"].iloc[0]
    std_speed = traj_data["std_speed"].iloc[0]
    ax.plot(traj_data["lon"], traj_data["lat"], linewidth=1)

    ax.text(
        0.05, 0.95, f"Avg: {avg_speed:.2f}\nStd: {std_speed:.2f}",
        transform=ax.transAxes,  # coordinates relative to the axes
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    ax.set_title(f"Trajectory {traj_id}", fontsize=10)
    ax.axis("off")
    
    # Hide any unused axes if chunk < 20
    ax.legend()
    
plt.tight_layout()
plt.show()