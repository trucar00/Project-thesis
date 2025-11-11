import pandas as pd
import folium
import pyarrow.parquet as pq
import numpy as np

df = pd.read_parquet('Processed_AIS/Parquets/2024-01-01.parquet', engine='pyarrow')
#df1 = pd.read_parquet('Processed_AIS/Parquets/2024-01-01.parquet', engine='pyarrow')
#df2 = pd.read_parquet('Processed_AIS/Parquets/2024-01-02.parquet', engine='pyarrow')
#df3 = pd.read_parquet('Processed_AIS/Parquets/2024-01-03.parquet', engine='pyarrow')
#df4 = pd.read_parquet('Processed_AIS/Parquets/2024-01-04.parquet', engine='pyarrow')
#df = pd.concat([df1, df2, df3, df4])

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

unique_mmsi = df["mmsi"].drop_duplicates()
print(unique_mmsi.shape)

vikanoy = df.loc[df["mmsi"] == 257700000]

vikanoy_sorted = vikanoy.sort_values(by="date_time_utc")
#print(vikanoy_sorted[["lon", "lat"]].head())

# Resampling
vikanoy_sorted = vikanoy_sorted.set_index("date_time_utc")
vikanoy_resampled = vikanoy_sorted.resample("1min", origin="start").first()

#print(vikanoy_resampled[["lon", "lat", "speed"]].head(100))

# Remove instances where the vessel has statyed still for a long period of time

def standStill(df, speed_threshold=0.3, min_duration=10): # Slow to iterate over rows, use pandas operations
    #df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    #df = df.sort_values(by=["mmsi", "date_time_utc"])

    count = 0
    drop = []
    for index, row in df.iterrows():
        #print(index)
        if row["speed"] < speed_threshold:
            count += 1
            drop.append(index)
        else:
            drop = []
            count = 0
        if count > min_duration:
            df = df.drop(drop)
            drop = []
        else:
            continue

    return df

#print(vikanoy_resampled.shape)
#test = standStill(vikanoy_resampled)
#print(test.shape)
#print(type(vikanoy_resampled))

def remove_stationary(df, speed_threshold=0.3, min_duration="10min"):
    df["stationary"] = df["speed"] < speed_threshold
    df["grp"] = (df["stationary"] != df["stationary"].shift()).cumsum()

    #print(df[df["stationary"]].groupby("grp"))

    drop = []
    for _, g in df[df["stationary"]].groupby("grp"):
        #print(g)
        #print(g.index)
        duration = g.index.max() - g.index.min()
        print(type(duration))
        if duration > pd.Timedelta(min_duration):
            drop.append(g.index)
        #print(duration)
    #print(pd.Index(drop))
    df = df.drop(pd.Index(np.concatenate(drop)))
    return df.drop(columns=["stationary", "grp"])

print(vikanoy_resampled.head())
boom = remove_stationary(vikanoy_resampled)
#print(boom.shape, vikanoy_resampled.shape)
print(boom.head(50))



# Plotting on a map
m = folium.Map(location=[vikanoy_sorted['lat'].iloc[0], vikanoy_sorted['lon'].iloc[0]], zoom_start=12)

# Add markers for each position
for i, row in vikanoy_sorted.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="red").add_to(m)

for i, row in boom.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="blue").add_to(m)

m.save("Maps/resampled_track.html")
m

""" nan_in_df = df.isnull().sum().sum()
print(df.head(10))
print(nan_in_df)

for column in df.columns:
    check_nan = df[column].isnull().sum().sum()
    print("There are ", check_nan, " in column ", column) """