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

df = pd.read_parquet("Processed_AIS/Concatenated/2024-01.parquet", engine="pyarrow")
df2 = pd.read_csv("Processed_AIS/TrajLonger2h/2024-01.csv")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df2["date_time_utc"] = pd.to_datetime(df2["date_time_utc"])
df.sort_values(by="date_time_utc")
df2.sort_values(by="date_time_utc")
