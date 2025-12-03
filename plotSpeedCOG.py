import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#df = pd.read_csv("Processed_AIS/Resampled/2024-02.csv")
df = pd.read_csv("newFeats.csv")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

print(df["trajectory_id"].unique(), df["trajectory_id"].nunique())

# Loop over trajectories
for _, d in df.groupby("trajectory_id"):
    mmsi = d["mmsi"].iloc[0]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex=True, gridspec_kw={'hspace': 0.15})

    # Plot speed
    ax1.plot(d["date_time_utc"], d["speed"], label="Speed", color="tab:green")
    ax1.set_ylabel("Speed [knots]")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(d["date_time_utc"], d["cog"], label="COG [°]", color="tab:blue")
    ax2.set_ylabel("COG [°]")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot course over ground
    ax3.plot(d["date_time_utc"], d["del_cog"], label="del COG", color="tab:orange")
    ax3.set_ylabel("del COG")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    ax4.plot(d["date_time_utc"], d["rot"], label="ROT", color="tab:red")
    ax4.set_ylabel("ROT")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    # X-axis formatting
    ax4.set_xlabel("Time (UTC)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))  # e.g. 14:30 on 12-Jan
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    fig.suptitle(f"MMSI: {mmsi} | Trajectory {d['trajectory_id'].iloc[0]}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
