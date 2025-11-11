import pandas as pd
import folium

df = pd.read_csv("Processed_AIS/Resampled/2024-02.csv")

#print(df["trajectory_id"].unique())

df1 = df.loc[df["trajectory_id"] == "250493000-1"]
df2 = df.loc[df["trajectory_id"] == "257005840-0"]
df3 = df.loc[df["trajectory_id"] == "257006840-0"] 
df4 = df.loc[df["trajectory_id"] == "257048070-0"] 

# Plotting on a map
m = folium.Map(location=[df['lat'].iloc[0], df['lon'].iloc[0]], zoom_start=12)

# Add markers for each position
for i, row in df1.iterrows():
    
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="red").add_to(m)

for i, row in df2.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="blue").add_to(m)

for i, row in df3.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="orange").add_to(m)

for i, row in df4.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="green").add_to(m)

#folium.CircleMarker(location=[69.65481, 18.970306], radius=2, color="black").add_to(m)


m.save("Maps/concat_resample_feb.html")
m

