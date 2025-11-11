import os
import dataProcessing
from time import time

# Extract the first 5 days of each month to catch all seasons

start = time()

for month in range(1,13):
    for day in range(1,6):
        filepath = f"Z:date_utc=2024-{month:02d}-{day:02d}"
        if os.path.exists(filepath):
            for entry in os.scandir(filepath):
                if entry.is_file() and entry.name.endswith(".parquet"):
                    print("Processing file: ", entry.path)
                    dataProcessing.readFilterSave(entry.path, f"Processed_AIS/Parquets/2024-{month:02d}-{day:02d}.parquet")
                    
        else:
            print("Missing: ", filepath)

end = time()
print("Done! It took: ", (end-start)/60, " minutes.")
