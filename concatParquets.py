import pandas as pd

for month in range(2,13):
    dfs = []
    for day in range(1,6):
        filepath = f"Processed_AIS/Parquets/2024-{month:02d}-{day:02d}.parquet"
        print(f"Concat of 2024-{month:02d}")
        df = pd.read_parquet(filepath, engine="pyarrow")
        dfs.append(df)

    concat_df = pd.concat(dfs, ignore_index=True)

    concat_df.to_parquet(f"Processed_AIS/Concatenated/2024-{month:02d}.parquet")


#print(concat_df.head())
#print(concat_df.tail())