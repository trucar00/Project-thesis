import pandas as pd
from time import time

# --- Concatenates the filtered parquet files ---

def main():
    start = time()
    print("Concatenating first five days of each month into common parquet file.")
    for month in range(1,13):
        dfs = []
        for day in range(1,6):
            filepath = f"Processed_AIS/Parquets/2024-{month:02d}-{day:02d}.parquet"
            print(f"Concat of 2024-{month:02d}")
            df = pd.read_parquet(filepath, engine="pyarrow")
            dfs.append(df)

        concat_df = pd.concat(dfs, ignore_index=True)

        concat_df.to_parquet(f"Processed_AIS/Concatenated/2024-{month:02d}.parquet")

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")
    
if __name__ == "__main__":
    main()