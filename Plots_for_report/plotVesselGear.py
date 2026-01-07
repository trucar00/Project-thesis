import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../Processed_AIS/Resampled3h/2024-03.csv")

df_v = df[df["ship_name"] == "HARVEST"].copy()

print(df_v["trajectory_id"].unique(), df_v["trajectory_id"].nunique())

df_v["date_time_utc"] = pd.to_datetime(df_v["date_time_utc"])
df_v = df_v.sort_values("date_time_utc")

for tid, d in df_v.groupby("trajectory_id"):
    print(tid)
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].plot(df_v["lon"], df_v["lat"])
    ax[1].plot(d["lon"], d["lat"]) # SUbtrajectory
    plt.show()


# TRAWL havfisk.no/en/fleet: 
# - 01, VESTTIND: 259683000 GOOD
# - 01, KONGSFJORD: 257089790
# - 01, GADUS POSEIDON: 258778000
# - 02, GRANIT: 257123000

# LINE ervikgroup
# - 03, FROYANES JUNIOR: 257489000
# - 03, BJORNHAUG: 258319000, ish GOOD
# - 07, VESTKAPP: 257487000
# - 09, ROLF ASBJOERN: 257133000, ish GOOD

# PURSE SEINE https://www.fiskebat.no/medlemsfartoy
# - 02, HAVSKJER: 257622000,
# - 03, SVANAUG ELISE: 258773000
# STEINEVIK: 259132000,
# LIAFJORD: 259026900 
# INGRID MAJALA: 257888000, good
# HARVEST: 257437000, good

# TODO:
# Plot the trajectories after one another? That creates gaps.
# Add some kind of ocean effect / map behind, add to report