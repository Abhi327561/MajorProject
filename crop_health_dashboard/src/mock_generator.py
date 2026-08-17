import os
import json
import pandas as pd
from datetime import datetime

def generate_mock_geojson(output_path):
    """Generates a GeoJSON file with 5 fictional agricultural fields in a cluster."""
    geojson_data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F01",
                    "field_name": "North Wheat Field",
                    "area_ha": 12.5
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [77.2050, 28.6150],
                        [77.2075, 28.6150],
                        [77.2075, 28.6170],
                        [77.2050, 28.6170],
                        [77.2050, 28.6150]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F02",
                    "field_name": "West Ridge Field",
                    "area_ha": 8.3
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [77.2010, 28.6130],
                        [77.2030, 28.6130],
                        [77.2030, 28.6148],
                        [77.2010, 28.6148],
                        [77.2010, 28.6130]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F03",
                    "field_name": "East Maize Field",
                    "area_ha": 15.1
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [77.2110, 28.6130],
                        [77.2135, 28.6130],
                        [77.2135, 28.6150],
                        [77.2110, 28.6150],
                        [77.2110, 28.6130]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F04",
                    "field_name": "South Barley Field",
                    "area_ha": 10.4
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [77.2050, 28.6090],
                        [77.2075, 28.6090],
                        [77.2075, 28.6110],
                        [77.2050, 28.6110],
                        [77.2050, 28.6090]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "field_id": "F05",
                    "field_name": "Central Clover Field",
                    "area_ha": 6.8
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [77.2050, 28.6120],
                        [77.2080, 28.6120],
                        [77.2080, 28.6140],
                        [77.2050, 28.6140],
                        [77.2050, 28.6120]
                    ]]
                }
            }
        ]
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=4)
    print(f"Mock GeoJSON successfully created at: {output_path}")

def generate_mock_csv(output_path):
    """Generates a CSV file with 12 monthly NDVI observations for each of the 5 fields,

    incorporating realistic cycles, anomalies, cloud cover, and different health classes.
    """
    
    # Month list from Jan 2026 to Dec 2026
    months = [f"2026-{m:02d}-15" for m in range(1, 13)]
    
    records = []
    
    # Scenario definitions
    # F01: Healthy & Stable
    # F02: Poor/Fallow/Low health
    # F03: Improving / Growing crop cycle
    # F04: Stress Event (sudden pest drop in July/August)
    # F05: Moderate & Variable
    
    for month_idx, date_str in enumerate(months):
        month_num = month_idx + 1
        
        # Determine cloud cover (realistic percentages: summer/monsoon months (Jul, Aug, Sep) have more clouds)
        if month_num in [6, 7, 8, 9]:
            cloud_cover = float(round(15.0 + (month_num % 3) * 12.5, 1))  # higher clouds: 15-40%
        else:
            cloud_cover = float(round(1.5 + (month_num % 4) * 2.2, 1))    # lower clouds: 1-10%
        
        # Field 1: North Wheat Field (Healthy & Stable)
        # Seasonal curve peaks around Mar/Apr, then again in Oct/Nov for double cropping
        if month_num in [2, 3, 4, 9, 10, 11]:
            ndvi_mean = float(round(0.72 + (month_num % 3) * 0.03, 2))
        else:
            ndvi_mean = float(round(0.62 - (month_num % 3) * 0.02, 2))
        ndvi_median = float(round(ndvi_mean - 0.01, 2))
        records.append({
            "field_id": "F01",
            "acquisition_date": date_str,
            "ndvi_mean": ndvi_mean,
            "ndvi_median": ndvi_median,
            "cloud_cover": cloud_cover
        })
        
        # Field 2: West Ridge Field (Poor/Fallow)
        # Consistently low NDVI, representing soil, weed cover, or poorly irrigated crop
        ndvi_mean_f2 = float(round(0.18 + (month_num % 4) * 0.02, 2))
        ndvi_median_f2 = float(round(ndvi_mean_f2 - 0.01, 2))
        records.append({
            "field_id": "F02",
            "acquisition_date": date_str,
            "ndvi_mean": ndvi_mean_f2,
            "ndvi_median": ndvi_median_f2,
            "cloud_cover": cloud_cover
        })
        
        # Field 3: East Maize Field (Improving / Growing crop)
        # Starts low in Jan (0.20) and climbs steadily as crop grows, peaking in Aug/Sep (0.75), then drops after harvest
        if month_num <= 4:
            ndvi_mean_f3 = float(round(0.22 + month_num * 0.05, 2))
        elif month_num <= 8:
            ndvi_mean_f3 = float(round(0.42 + (month_num - 4) * 0.08, 2))
        elif month_num <= 10:
            ndvi_mean_f3 = float(round(0.74 - (month_num - 8) * 0.06, 2))
        else:
            ndvi_mean_f3 = float(round(0.25 - (month_num - 10) * 0.03, 2))
        ndvi_median_f3 = float(round(ndvi_mean_f3 - 0.02, 2))
        records.append({
            "field_id": "F03",
            "acquisition_date": date_str,
            "ndvi_mean": ndvi_mean_f3,
            "ndvi_median": ndvi_median_f3,
            "cloud_cover": cloud_cover
        })
        
        # Field 4: South Barley Field (Stress Event in July/August)
        # Normally healthy (0.65-0.70) but suffers a massive drop in July (e.g. down to 0.38, drop of 0.32 > 0.20 threshold)
        # Fictional stress reason: localized insect attack / drought. Slowly recovers in Oct/Nov.
        if month_num in [1, 2, 3, 4, 5, 6]:
            ndvi_mean_f4 = float(round(0.68 + (month_idx % 3) * 0.02, 2))
        elif month_num == 7:
            # Significant drop! From 0.70 (June) to 0.38 (July)
            ndvi_mean_f4 = 0.38
        elif month_num == 8:
            # Continues low
            ndvi_mean_f4 = 0.35
        elif month_num == 9:
            # Moderate recovery starts
            ndvi_mean_f4 = 0.45
        else:
            # Higher recovery
            ndvi_mean_f4 = float(round(0.55 + (month_num % 2) * 0.05, 2))
        
        ndvi_median_f4 = float(round(ndvi_mean_f4 - 0.01, 2))
        records.append({
            "field_id": "F04",
            "acquisition_date": date_str,
            "ndvi_mean": ndvi_mean_f4,
            "ndvi_median": ndvi_median_f4,
            "cloud_cover": cloud_cover
        })
        
        # Field 5: Central Clover Field (Moderate & Variable)
        # Stays mostly in the moderate range (0.35 - 0.58)
        ndvi_mean_f5 = float(round(0.42 + ((month_num * 7) % 5) * 0.03, 2))
        ndvi_median_f5 = float(round(ndvi_mean_f5 + 0.01, 2))
        records.append({
            "field_id": "F05",
            "acquisition_date": date_str,
            "ndvi_mean": ndvi_mean_f5,
            "ndvi_median": ndvi_median_f5,
            "cloud_cover": cloud_cover
        })
        
    df = pd.DataFrame(records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Mock NDVI time-series CSV successfully created at: {output_path}")

if __name__ == "__main__":
    # Base paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    geojson_out = os.path.join(project_dir, "data", "mock_fields.geojson")
    csv_out = os.path.join(project_dir, "data", "mock_ndvi_timeseries.csv")
    
    generate_geojson_out = os.path.join(project_dir, "data", "mock_fields.geojson")
    generate_mock_geojson(geojson_out)
    generate_mock_csv(csv_out)
