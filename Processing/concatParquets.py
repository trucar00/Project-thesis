import pandas as pd
from time import time

# --- Concatenates the filtered parquet files ---

def main(months, days, filtered_path, concat_path):
    start = time()
    print("Concatenating first five days of each month into common parquet file.")
    for month in range(1, months+1):
        dfs = []
        for day in range(1, days+1):
            filepath = f"{filtered_path}2024-{month:02d}-{day:02d}.parquet"
            print(f"Concat of 2024-{month:02d}")
            df = pd.read_parquet(filepath, engine="pyarrow")
            dfs.append(df)

        concat_df = pd.concat(dfs, ignore_index=True)

        concat_df.to_parquet(f"{concat_path}{month:02d}.parquet")

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")
    
if __name__ == "__main__":
    main()