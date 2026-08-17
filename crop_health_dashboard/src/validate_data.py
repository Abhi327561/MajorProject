import os
import json
import csv
from datetime import datetime

# Import thresholds from config if available, otherwise define defaults
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    NDVI_POOR_THRESHOLD = config.NDVI_POOR_THRESHOLD
    NDVI_HEALTHY_THRESHOLD = config.NDVI_HEALTHY_THRESHOLD
    ANOMALY_DROP_THRESHOLD = config.ANOMALY_DROP_THRESHOLD
    MAX_CLOUD_COVER = config.MAX_CLOUD_COVER
except ImportError:
    NDVI_POOR_THRESHOLD = 0.30
    NDVI_HEALTHY_THRESHOLD = 0.60
    ANOMALY_DROP_THRESHOLD = 0.20
    MAX_CLOUD_COVER = 20.0

def validate_datasets():
    print("=" * 60)
    print("STARTING DATA VALIDATION (STANDARD LIBRARY VERSION)")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    geojson_path = os.path.abspath(os.path.join(project_dir, "data", "mock_fields.geojson"))
    csv_path = os.path.abspath(os.path.join(project_dir, "data", "mock_ndvi_timeseries.csv"))
    
    print(f"GeoJSON Path: {geojson_path}")
    print(f"CSV Path:     {csv_path}")
    
    errors = []
    
    # 1. File Existence Check
    if not os.path.exists(geojson_path):
        errors.append(f"GeoJSON file does not exist at {geojson_path}")
    if not os.path.exists(csv_path):
        errors.append(f"CSV file does not exist at {csv_path}")
        
    if errors:
        print("\n[FAILED] Initial existence check failed:")
        for err in errors:
            print(f"  - {err}")
        return False
        
    print("\n[SUCCESS] Both mock files exist.")
    
    # 2. Parse GeoJSON (Standard json library)
    geojson_fields = {}
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            
        if geojson_data.get("type") != "FeatureCollection":
            errors.append("GeoJSON type is not 'FeatureCollection'")
            
        features = geojson_data.get("features", [])
        print(f"[SUCCESS] GeoJSON loaded successfully. Features count: {len(features)}")
        
        for idx, feature in enumerate(features):
            if feature.get("type") != "Feature":
                errors.append(f"Feature at index {idx} type is not 'Feature'")
                continue
                
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            
            # Check properties
            field_id = props.get("field_id")
            field_name = props.get("field_name")
            area_ha = props.get("area_ha")
            
            if not field_id:
                errors.append(f"Feature at index {idx} is missing 'field_id'")
                continue
            if not field_name:
                errors.append(f"Feature '{field_id}' is missing 'field_name'")
            if area_ha is None or not isinstance(area_ha, (int, float)):
                errors.append(f"Feature '{field_id}' is missing or has invalid 'area_ha'")
                
            # Check geometry
            geom_type = geom.get("type")
            coords = geom.get("coordinates")
            if geom_type != "Polygon":
                errors.append(f"Feature '{field_id}' geometry is not a 'Polygon' (got '{geom_type}')")
            elif not isinstance(coords, list) or len(coords) == 0:
                errors.append(f"Feature '{field_id}' coordinates are missing or invalid")
            else:
                # Check closed loop for first polygon ring
                ring = coords[0]
                if len(ring) < 4:
                    errors.append(f"Feature '{field_id}' polygon ring has too few points (min 4)")
                if ring[0] != ring[-1]:
                    errors.append(f"Feature '{field_id}' polygon is not closed (first point {ring[0]} != last point {ring[-1]})")
                    
            geojson_fields[field_id] = {
                "field_name": field_name,
                "area_ha": area_ha,
                "coordinates": coords
            }
            
    except Exception as e:
        errors.append(f"Could not parse GeoJSON file: {e}")
        
    # 3. Parse CSV (Standard csv library)
    csv_records = []
    csv_fields = set()
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            # Validate Headers
            required_headers = ["field_id", "acquisition_date", "ndvi_mean", "ndvi_median", "cloud_cover"]
            for header in required_headers:
                if header not in reader.fieldnames:
                    errors.append(f"CSV is missing required header: '{header}'")
                    
            if not errors:
                for row_idx, row in enumerate(reader, start=2):
                    field_id = row.get("field_id")
                    date_str = row.get("acquisition_date")
                    ndvi_mean_str = row.get("ndvi_mean")
                    ndvi_median_str = row.get("ndvi_median")
                    cloud_cover_str = row.get("cloud_cover")
                    
                    csv_fields.add(field_id)
                    
                    # Type checking & validation
                    try:
                        # Validate Date
                        datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        errors.append(f"Row {row_idx}: date '{date_str}' is not in YYYY-MM-DD format")
                        
                    try:
                        ndvi_mean = float(ndvi_mean_str)
                        if not (-1.0 <= ndvi_mean <= 1.0):
                            errors.append(f"Row {row_idx}: ndvi_mean {ndvi_mean} is out of bounds [-1, 1]")
                    except ValueError:
                        errors.append(f"Row {row_idx}: ndvi_mean '{ndvi_mean_str}' is not a valid float")
                        ndvi_mean = None
                        
                    try:
                        ndvi_median = float(ndvi_median_str)
                        if not (-1.0 <= ndvi_median <= 1.0):
                            errors.append(f"Row {row_idx}: ndvi_median {ndvi_median} is out of bounds [-1, 1]")
                    except ValueError:
                        errors.append(f"Row {row_idx}: ndvi_median '{ndvi_median_str}' is not a valid float")
                        ndvi_median = None
                        
                    try:
                        cloud_cover = float(cloud_cover_str)
                        if not (0.0 <= cloud_cover <= 100.0):
                            errors.append(f"Row {row_idx}: cloud_cover {cloud_cover} is out of bounds [0, 100]")
                    except ValueError:
                        errors.append(f"Row {row_idx}: cloud_cover '{cloud_cover_str}' is not a valid float")
                        cloud_cover = None
                        
                    csv_records.append({
                        "field_id": field_id,
                        "acquisition_date": date_str,
                        "ndvi_mean": ndvi_mean,
                        "ndvi_median": ndvi_median,
                        "cloud_cover": cloud_cover,
                        "row_idx": row_idx
                    })
        print(f"[SUCCESS] CSV loaded successfully. Records count: {len(csv_records)}")
    except Exception as e:
        errors.append(f"Could not parse CSV file: {e}")
        
    if errors:
        print("\n[FAILED] Structural or parsing validation failed:")
        for err in errors:
            print(f"  - {err}")
        return False
        
    print("[SUCCESS] Schemas and formats are valid.")

    # 4. Check Internal Consistency (Referential Integrity & Quantities)
    geojson_ids = set(geojson_fields.keys())
    
    print(f"\nFields in GeoJSON: {sorted(list(geojson_ids))}")
    print(f"Fields in CSV:     {sorted(list(csv_fields))}")
    
    if geojson_ids != csv_fields:
        missing_in_csv = geojson_ids - csv_fields
        missing_in_geojson = csv_fields - geojson_ids
        if missing_in_csv:
            errors.append(f"Fields present in GeoJSON but missing in CSV: {missing_in_csv}")
        if missing_in_geojson:
            errors.append(f"Fields present in CSV but missing in GeoJSON: {missing_in_geojson}")
    else:
        print("[SUCCESS] Referential integrity matches perfectly between files.")

    # Check temporal record count per field (6-12 observations)
    for fid in geojson_ids:
        field_observations = [r for r in csv_records if r["field_id"] == fid]
        count = len(field_observations)
        print(f"  - Field {fid} ({geojson_fields[fid]['field_name']}) has {count} temporal records.")
        if count < 6 or count > 12:
            errors.append(f"Field {fid} has {count} observations, which is outside the required range [6, 12]")

    # 5. Check Scenario Validation (Poor, Healthy, Moderate, Anomaly, Cloud Filter)
    print("\nChecking logical scenario conditions:")
    
    healthy_count = 0
    moderate_count = 0
    poor_count = 0
    cloudy_count = 0
    anomalies_detected = 0
    
    # Sort records by field and date for sequence analysis
    sorted_records = sorted(csv_records, key=lambda x: (x["field_id"], x["acquisition_date"]))
    
    # Track historical sequence per field to detect drops
    field_history = {}
    for r in sorted_records:
        fid = r["field_id"]
        ndvi = r["ndvi_mean"]
        clouds = r["cloud_cover"]
        date_str = r["acquisition_date"]
        
        if ndvi is None or clouds is None:
            continue
            
        # Count classifications (before filtering cloud cover for raw check, or after? Let's check raw values)
        if ndvi >= NDVI_HEALTHY_THRESHOLD:
            healthy_count += 1
        elif ndvi < NDVI_POOR_THRESHOLD:
            poor_count += 1
        else:
            moderate_count += 1
            
        if clouds > MAX_CLOUD_COVER:
            cloudy_count += 1
            
        # Anomaly detection (drop between consecutive dates)
        if fid not in field_history:
            field_history[fid] = []
            
        if len(field_history[fid]) > 0:
            prev_r = field_history[fid][-1]
            prev_ndvi = prev_r["ndvi_mean"]
            drop = prev_ndvi - ndvi
            if drop >= ANOMALY_DROP_THRESHOLD:
                anomalies_detected += 1
                print(f"    * Detected anomaly on Field {fid} at {date_str}: dropped by {drop:.2f} (from {prev_ndvi:.2f} to {ndvi:.2f})")
                
        field_history[fid].append(r)
        
    print(f"  - Healthy readings (NDVI >= {NDVI_HEALTHY_THRESHOLD}): {healthy_count}")
    print(f"  - Moderate readings ({NDVI_POOR_THRESHOLD} <= NDVI < {NDVI_HEALTHY_THRESHOLD}): {moderate_count}")
    print(f"  - Poor readings (NDVI < {NDVI_POOR_THRESHOLD}): {poor_count}")
    print(f"  - High cloud cover readings (> {MAX_CLOUD_COVER}%): {cloudy_count}")
    print(f"  - Total sudden drops (>= {ANOMALY_DROP_THRESHOLD}): {anomalies_detected}")
    
    if healthy_count == 0:
        errors.append("No healthy crop scenario found in data.")
    if poor_count == 0:
        errors.append("No poor crop/soil scenario found in data.")
    if moderate_count == 0:
        errors.append("No moderate crop scenario found in data.")
    if cloudy_count == 0:
        errors.append("No high cloud cover scenario found in data.")
    if anomalies_detected == 0:
        errors.append("No sudden health drops (anomalies) found in data.")
        
    print("=" * 60)
    if errors:
        print("[FAIL] Validation completed with errors:")
        for err in errors:
            print(f"  - {err}")
        print("=" * 60)
        return False
    else:
        print("[PASS] All validations passed! Fictional scenarios and datasets are internally consistent.")
        print("=" * 60)
        return True

if __name__ == "__main__":
    validate_datasets()
