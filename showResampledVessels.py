import pandas as pd
import folium

#df = pd.read_csv("Processed_AIS/Resampled2/2024-01.csv")
df = pd.read_csv("Featureset/3h_1h_step/2024-01.csv")

#print(df["trajectory_id"].unique())

df1 = df.loc[df["trajectory_id"] == "231867000-11"]
df2 = df.loc[df["trajectory_id"] == "231867000-10"]
df3 = df.loc[df["trajectory_id"] == "231867000-9"] 
df4 = df.loc[df["trajectory_id"] == "231867000-8"]
df5 = df.loc[df["trajectory_id"] == "231867000-6"]

# Plotting on a map
m = folium.Map(location=[df['lat'].iloc[0], df['lon'].iloc[0]], zoom_start=12)

# Add markers for each position
for i, row in df1.iterrows():
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=5, color="red").add_to(m)

for i, row in df2.iterrows():
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=4, color="blue").add_to(m)

for i, row in df3.iterrows():
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=3, color="orange").add_to(m)

for i, row in df4.iterrows():
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="green").add_to(m)

for i, row in df5.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=1, color="yellow").add_to(m)

#folium.CircleMarker(location=[73.65999, 18.254028], radius=5, color="black").add_to(m)



m.save("Maps/train_traj.html")
m

# 18.254028 73.65999 -> 18.833889 73.66693
# 18.477568 73.67059 -> 18.95022 73.67231
# 18.590794 73.67293 -> 19.064718 73.679115
