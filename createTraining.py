import pandas as pd
import matplotlib.pyplot as plt

# Create 1h chunks for each trajectory_id. Start from first timestamp for that trajectory_id and then divide up into as many 1h intervals as possible. 
# If at some point the next one hour interval is outside the timestamp for that trajectory_id, just throw it away (likely close to harbor anyways).

df = pd.read_csv("Processed_AIS/Resampled/2024-01.csv")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df = df.sort_values(["trajectory_id", "date_time_utc"])
chunks = []

for tid, group in df.groupby("trajectory_id"):
        start_time = group["date_time_utc"].iloc[0]
        end_time = group["date_time_utc"].iloc[-1]

        # Determine how many full 1-hour windows fit into this trajectory
        total_hours = int((end_time - start_time).total_seconds() // 3600)

        for i in range(total_hours):
            t0 = start_time + pd.Timedelta(hours=i)
            t1 = t0 + pd.Timedelta(hours=1) # may need more hours

            # Extract data within this 1-hour window
            chunk = group[(group["date_time_utc"] >= t0) & (group["date_time_utc"] < t1)].copy()
            if len(chunk) > 1:
                chunk["chunk_id"] = f"{tid}_{i:03d}"
                chunks.append(chunk)


""" df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df = df.sort_values(["trajectory_id", "date_time_utc"])

# Get first and last position per trajectory
first_points = df.groupby("trajectory_id").first().reset_index()
last_points = df.groupby("trajectory_id").last().reset_index()

plt.figure(figsize=(10, 8))
plt.scatter(first_points["lon"], first_points["lat"], color="green", s=20, label="Start")
plt.scatter(last_points["lon"], last_points["lat"], color="red", s=20, label="End")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.title("Start (green) and End (red) Positions per Trajectory")
plt.grid(True)
plt.show() """