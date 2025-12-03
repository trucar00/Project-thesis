import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Featureset/3h_1h_step/2024-01.csv")

traj_ids = df["trajectory_id"].unique()
print(len(traj_ids))

n = 20

res = [traj_ids[i:i + n] for i in range(0, len(traj_ids), n)]

for idx, id_chunk in enumerate(res):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()  # make it easy to index
    
    for ax, traj_id in zip(axes, id_chunk):
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
    for ax in axes[len(id_chunk):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()