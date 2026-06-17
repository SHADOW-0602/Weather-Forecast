import os
import zipfile
import xarray as xr
import pandas as pd
import numpy as np

# Directory containing zip chunks
zip_dir = "sst_chunks"
extracted_dir = "sst_extracted"
os.makedirs(extracted_dir, exist_ok=True)

# 1. Load existing compiled records
records_dict = {}

daily_path = "ocean.csv"
monthly_path = "ocean_monthly.csv"

if os.path.exists(daily_path):
    try:
        df_existing = pd.read_csv(daily_path)
        for _, row in df_existing.iterrows():
            records_dict[str(row["Date"])] = float(row["SST_Mean_Celsius"])
        print(f"Loaded {len(df_existing)} existing records from {daily_path}")
    except Exception as e:
        print(f"Error reading {daily_path}: {e}")

if os.path.exists(monthly_path):
    try:
        df_existing = pd.read_csv(monthly_path)
        for _, row in df_existing.iterrows():
            d = str(row["Date"])
            if d not in records_dict:
                records_dict[d] = float(row["SST_Mean_Celsius"])
        print(f"Loaded records from {monthly_path}. Total unique records: {len(records_dict)}")
    except Exception as e:
        print(f"Error reading {monthly_path}: {e}")

# 2. Extract only new NetCDF files from zip archives
print("Extracting new NetCDF files from zip archives...")
zip_files = sorted([f for f in os.listdir(zip_dir) if f.endswith(".zip")])

for filename in zip_files:
    zip_path = os.path.join(zip_dir, filename)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check if zip contains any file we haven't processed yet
            needs_extraction = False
            for member in zip_ref.namelist():
                if member.endswith(".nc"):
                    # Parse date from filename to check if we already processed it
                    date_str = member.split("-")[0]
                    if len(date_str) >= 8 and date_str.isdigit():
                        date_val = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    elif len(date_str) == 6 and date_str.isdigit():
                        date_val = f"{date_str[:4]}-{date_str[4:6]}-01"
                    else:
                        continue
                    
                    if date_val not in records_dict:
                        needs_extraction = True
                        break
            
            if needs_extraction:
                print(f"Extracting {filename}...")
                for member in zip_ref.namelist():
                    if member.endswith(".nc"):
                        target_path = os.path.join(extracted_dir, member)
                        if not os.path.exists(target_path):
                            zip_ref.extract(member, extracted_dir)
            else:
                pass # Already processed all files in this archive
    except Exception as e:
        print(f"Error checking/extracting {filename}: {e}")

# 3. Process only new NetCDF files
print("Processing new NetCDF files...")
nc_files = sorted([f for f in os.listdir(extracted_dir) if f.endswith(".nc")])

new_records_added = 0
for f in nc_files:
    file_path = os.path.join(extracted_dir, f)
    # Parse date from filename
    date_str = f.split("-")[0]
    if len(date_str) >= 8 and date_str.isdigit():
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        date_val = f"{year}-{month:02d}-{day:02d}"
    elif len(date_str) == 6 and date_str.isdigit():
        year = int(date_str[:4])
        month = int(date_str[4:6])
        date_val = f"{year}-{month:02d}-01"
    else:
        continue
        
    # Skip if already in records_dict
    if date_val in records_dict:
        continue
        
    try:
        # Open dataset
        with xr.open_dataset(file_path) as ds:
            mean_sst_k = float(ds.analysed_sst.mean(skipna=True).values)
            if not np.isnan(mean_sst_k):
                mean_sst_c = mean_sst_k - 273.15
                records_dict[date_val] = mean_sst_c
                new_records_added += 1
                print(f"Processed new date {date_val}: {mean_sst_c:.2f} °C")
            else:
                print(f"Warning: {f} yielded NaN mean SST")
    except Exception as e:
        print(f"Error processing {f}: {e}")

# 4. Save compiled records
if new_records_added > 0 or not os.path.exists(daily_path):
    records_list = [{"Date": d, "SST_Mean_Celsius": s} for d, s in records_dict.items()]
    df = pd.DataFrame(records_list)
    df = df.sort_values("Date")
    df.to_csv(daily_path, index=False)
    print(f"Saved {len(df)} total daily/monthly records to {daily_path} (added {new_records_added} new records)")
else:
    print("No new records to add. Summary is up-to-date.")
