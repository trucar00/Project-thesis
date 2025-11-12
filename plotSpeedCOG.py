import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("Processed_AIS/Resampled/2024-02.csv")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

print(df["trajectory_id"].unique(), df["trajectory_id"].nunique())

# Loop over trajectories
for _, d in df.groupby("trajectory_id"):
    mmsi = d["mmsi"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0.15})

    # Plot speed
    ax1.plot(d["date_time_utc"], d["speed"], label="Speed [knots]", color="tab:blue")
    ax1.set_ylabel("Speed [knots]")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot course over ground
    ax2.plot(d["date_time_utc"], d["cog"], label="COG [°]", color="tab:orange")
    ax2.set_ylabel("COG [°]")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # X-axis formatting
    ax2.set_xlabel("Time (UTC)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))  # e.g. 14:30 on 12-Jan
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    fig.suptitle(f"MMSI: {mmsi} | Trajectory {d['trajectory_id'].iloc[0]}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# REMAININ OUTLIERS
# 257570500-0