import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import tifffile as t


# ============================================================
# PATHS
# ============================================================

SENTINEL_DIR = r"C:\Users\mpabh\OneDrive\Desktop\srcdata\sentinel2\AT"

MASK_DIR = r"C:\Users\mpabh\OneDrive\Desktop\AI4Boundaries\masks"

OUTPUT_DIR = r"C:\Users\mpabh\OneDrive\Desktop\MajorProject\model\outputs\ndvi"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND MATCHING FILES
# ============================================================

images = glob.glob(os.path.join(SENTINEL_DIR, "*.nc"))

print(f"Found Sentinel images: {len(images)}")


# Create mask lookup using filename
mask_files = glob.glob(
    os.path.join(MASK_DIR, "**", "*.tif"),
    recursive=True
)

mask_lookup = {
    os.path.splitext(os.path.basename(f))[0]: f
    for f in mask_files
}

print(f"Found masks: {len(mask_files)}")


# ============================================================
# PROCESS EACH SAMPLE
# ============================================================

all_records = []

for idx, image_path in enumerate(images):

    sample_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    if sample_name not in mask_lookup:
        print(f"[SKIP] No mask: {sample_name}")
        continue

    mask_path = mask_lookup[sample_name]

    print(
        f"[{idx + 1}/{len(images)}] "
        f"Processing {sample_name}"
    )

    # --------------------------------------------------------
    # Load Sentinel data
    # --------------------------------------------------------

    ds = xr.open_dataset(image_path)

    ndvi = ds["NDVI"].values
    times = ds["time"].values

    # Shape:
    # (time, 256, 256)

    # --------------------------------------------------------
    # Load mask
    # --------------------------------------------------------

    mask = t.imread(mask_path)

    # Mask shape:
    # (256, 256, 4)

    # Channel 3 = field ID
    field_ids = mask[:, :, 3]

    # --------------------------------------------------------
    # Valid field IDs
    # --------------------------------------------------------

    valid_ids = np.unique(
        field_ids[field_ids >= 0]
    )

    print(
        f"  Fields detected: {len(valid_ids)}"
    )

    # --------------------------------------------------------
    # Extract NDVI for every field/month
    # --------------------------------------------------------

    for field_id in valid_ids:

        field_pixels = field_ids == field_id

        for month_index, time_value in enumerate(times):

            values = ndvi[month_index][field_pixels]

            # Remove invalid NDVI
            values = values[
                np.isfinite(values)
            ]

            values = values[
                values > -1
            ]

            if len(values) == 0:
                median_ndvi = np.nan
                mean_ndvi = np.nan
            else:
                median_ndvi = float(
                    np.median(values)
                )

                mean_ndvi = float(
                    np.mean(values)
                )

            all_records.append({
                "sample": sample_name,
                "field_id": int(field_id),
                "date": str(
                    np.datetime_as_string(
                        time_value,
                        unit="D"
                    )
                ),
                "year": int(
                    str(time_value)[:4]
                ),
                "month": int(
                    str(time_value)[5:7]
                ),
                "median_ndvi": median_ndvi,
                "mean_ndvi": mean_ndvi,
                "pixel_count": len(values)
            })

    ds.close()


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(all_records)

output_csv = os.path.join(
    OUTPUT_DIR,
    "field_ndvi.csv"
)

df.to_csv(
    output_csv,
    index=False
)

print()
print("=" * 60)
print("NDVI EXTRACTION COMPLETE")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Samples: {df['sample'].nunique()}")
print(f"Fields: {df[['sample', 'field_id']].drop_duplicates().shape[0]}")
print(f"Output: {output_csv}")

print()
print("First 20 rows:")
print(df.head(20).to_string(index=False))