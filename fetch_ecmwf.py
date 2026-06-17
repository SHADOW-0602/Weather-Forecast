import os
import cdsapi

# Use the local configuration file instead of the global one
os.environ["CDSAPI_RC"] = ".cdsapirc"

# Create output directories if they don't exist
os.makedirs("sst_chunks", exist_ok=True)

dataset = "satellite-sea-surface-temperature"

# Define the years from 1980 to 2025
years = [str(y) for y in range(1980, 2026)]
months = [f"{m:02d}" for m in range(1, 13)]

client = cdsapi.Client()

for year in years:
    for month in months:
        output_filename = f"sst_chunks/sst_daily_{year}_{month}.zip"
        
        if os.path.exists(output_filename):
            continue
            
        print(f"Retrieving daily SST data for {year}-{month}...")
        request = {
            "variable": "all",
            "processinglevel": "level_4",
            "sensor_on_satellite": "combined_product",
            "version": "3_0",
            "temporal_resolution": "daily",
            "year": [year],
            "month": [month],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "area": [55, -135, 20, -60]  # Bounding box covering U.S. coastal waters
        }
        
        try:
            client.retrieve(dataset, request).download(output_filename)
            print(f"Successfully downloaded {output_filename}")
        except Exception as e:
            print(f"Error downloading {year}-{month}: {e}")
            break
    else:
        # Continue outer loop if inner loop completes without break
        continue
    # Break outer loop if inner loop encountered error and broke
    break

print("All downloads processed.")
