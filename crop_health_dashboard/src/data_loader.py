import os
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, Tuple
import sys

# Import config to get paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
    DEFAULT_GEOJSON_PATH = config.GEOJSON_PATH
    DEFAULT_CSV_PATH = config.CSV_PATH
except ImportError:
    # Fallback paths if import fails
    DEFAULT_GEOJSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mock_fields.geojson")
    DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mock_ndvi_timeseries.csv")

def load_field_boundaries(path: str = None) -> gpd.GeoDataFrame:
    """Loads the spatial boundaries (GeoJSON) of the fields.

    Args:
        path: Path to the GeoJSON file. Defaults to config.GEOJSON_PATH.

    Returns:
        A GeoPandas GeoDataFrame containing the field boundaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, invalid, or missing required columns.
    """
    file_path = path or DEFAULT_GEOJSON_PATH
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GeoJSON file not found at: {file_path}")
        
    try:
        gdf = gpd.read_file(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read GeoJSON file: {e}")
        
    if gdf.empty:
        raise ValueError(f"GeoJSON file is empty: {file_path}")
        
    # Validate structure
    required_cols = {'field_id', 'field_name', 'area_ha', 'geometry'}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise ValueError(f"GeoJSON is missing required columns: {sorted(list(missing))}")
        
    return gdf

def load_ndvi_data(path: str = None) -> pd.DataFrame:
    """Loads the NDVI time-series (CSV) data.

    Args:
        path: Path to the CSV file. Defaults to config.CSV_PATH.

    Returns:
        A Pandas DataFrame containing the NDVI time-series data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, invalid, or missing required columns.
    """
    file_path = path or DEFAULT_CSV_PATH
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NDVI CSV file not found at: {file_path}")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
        
    if df.empty:
        raise ValueError(f"NDVI CSV file is empty: {file_path}")
        
    # Validate structure
    required_cols = {'field_id', 'acquisition_date', 'ndvi_mean', 'ndvi_median', 'cloud_cover'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"NDVI CSV is missing required columns: {sorted(list(missing))}")
        
    # Convert acquisition_date to datetime type
    try:
        df['acquisition_date'] = pd.to_datetime(df['acquisition_date'], format='%Y-%m-%d', errors='raise')
    except Exception as e:
        raise ValueError(f"Failed to parse acquisition_date in YYYY-MM-DD format: {e}")
        
    return df

def validate_field_data(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Validates the GeoJSON structure, columns, and geometries.

    Args:
        gdf: The loaded field boundaries GeoDataFrame.

    Returns:
        A dictionary containing validation status ('success': bool), 'errors': list, and 'warnings': list.
    """
    report = {
        "success": True,
        "errors": [],
        "warnings": []
    }
    
    # 1. Column presence check
    required_cols = {'field_id', 'field_name', 'area_ha', 'geometry'}
    missing = required_cols - set(gdf.columns)
    if missing:
        report["errors"].append(f"Missing required columns: {sorted(list(missing))}")
        report["success"] = False
        return report # Critical structure issue
        
    # 2. Check for null values in critical attributes
    for col in ['field_id', 'field_name', 'area_ha']:
        null_count = gdf[col].isnull().sum()
        if null_count > 0:
            report["errors"].append(f"Column '{col}' contains {null_count} missing (null) values.")
            report["success"] = False
            
    # 3. Check for unique field_ids
    duplicate_ids = gdf['field_id'][gdf['field_id'].duplicated()].unique().tolist()
    if duplicate_ids:
        report["errors"].append(f"Duplicate field_ids found in GeoJSON: {duplicate_ids}")
        report["success"] = False
        
    # 4. Geometry validity check
    invalid_geoms = gdf[~gdf.geometry.is_valid]
    if not invalid_geoms.empty:
        report["errors"].append(f"Found {len(invalid_geoms)} invalid geometries. IDs: {invalid_geoms['field_id'].tolist()}")
        report["success"] = False
        
    # 5. Check coordinates (warn if empty/none)
    empty_geoms = gdf[gdf.geometry.is_empty]
    if not empty_geoms.empty:
        report["errors"].append(f"Found empty geometries. IDs: {empty_geoms['field_id'].tolist()}")
        report["success"] = False
        
    return report

def validate_ndvi_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates CSV formatting, range restrictions, duplicates, and missing values.

    Args:
        df: The loaded NDVI time-series DataFrame.

    Returns:
        A dictionary containing validation status ('success': bool), 'errors': list, and 'warnings': list.
    """
    report = {
        "success": True,
        "errors": [],
        "warnings": []
    }
    
    # 1. Check required columns
    required_cols = {'field_id', 'acquisition_date', 'ndvi_mean', 'ndvi_median', 'cloud_cover'}
    missing = required_cols - set(df.columns)
    if missing:
        report["errors"].append(f"Missing required columns: {sorted(list(missing))}")
        report["success"] = False
        return report
        
    # 2. Check for missing values (nulls)
    null_summary = df[list(required_cols)].isnull().sum()
    for col, count in null_summary.items():
        if count > 0:
            report["errors"].append(f"Column '{col}' contains {count} missing (null) values.")
            report["success"] = False
            
    # 3. Range checking for NDVI values (Normally between -1.0 and 1.0)
    # Filter rows with non-null values for range checking
    valid_ndvi_mean = df['ndvi_mean'].dropna()
    out_of_bounds_mean = df[(df['ndvi_mean'] < -1.0) | (df['ndvi_mean'] > 1.0)]
    if not out_of_bounds_mean.empty:
        report["errors"].append(
            f"Found {len(out_of_bounds_mean)} row(s) where ndvi_mean is outside the [-1.0, 1.0] range. "
            f"Row indices: {out_of_bounds_mean.index.tolist()}"
        )
        report["success"] = False
        
    out_of_bounds_median = df[(df['ndvi_median'] < -1.0) | (df['ndvi_median'] > 1.0)]
    if not out_of_bounds_median.empty:
        report["errors"].append(
            f"Found {len(out_of_bounds_median)} row(s) where ndvi_median is outside the [-1.0, 1.0] range. "
            f"Row indices: {out_of_bounds_median.index.tolist()}"
        )
        report["success"] = False
        
    # 4. Check cloud cover percentages [0.0, 100.0]
    out_of_bounds_cloud = df[(df['cloud_cover'] < 0.0) | (df['cloud_cover'] > 100.0)]
    if not out_of_bounds_cloud.empty:
        report["errors"].append(
            f"Found {len(out_of_bounds_cloud)} row(s) where cloud_cover is outside [0.0, 100.0] range. "
            f"Row indices: {out_of_bounds_cloud.index.tolist()}"
        )
        report["success"] = False
        
    # 5. Detect duplicate field/date observations
    duplicates = df[df.duplicated(subset=['field_id', 'acquisition_date'], keep=False)]
    if not duplicates.empty:
        dup_info = duplicates.groupby(['field_id', 'acquisition_date']).size().reset_index(name='count')
        dup_details = []
        for _, row in dup_info.iterrows():
            d_val = row['acquisition_date']
            d_str = d_val.strftime('%Y-%m-%d') if hasattr(d_val, 'strftime') else str(d_val)
            dup_details.append(f"Field: {row['field_id']} on {d_str} (Count: {row['count']})")
        report["errors"].append(f"Duplicate observations detected for the same field on the same date: {dup_details}")
        report["success"] = False
        
    return report

def validate_field_ids(gdf: gpd.GeoDataFrame, df: pd.DataFrame) -> Dict[str, Any]:
    """Validates referential integrity between spatial and temporal datasets.

    Args:
        gdf: GeoDataFrame containing field boundaries.
        df: DataFrame containing NDVI records.

    Returns:
        A dictionary containing validation status ('success': bool), 'errors': list, and 'warnings': list.
    """
    report = {
        "success": True,
        "errors": [],
        "warnings": []
    }
    
    geojson_ids = set(gdf['field_id'].unique())
    csv_ids = set(df['field_id'].unique())
    
    # 1. Every NDVI field_id must exist in GeoJSON boundaries
    unmatched_csv_ids = csv_ids - geojson_ids
    if unmatched_csv_ids:
        report["errors"].append(
            f"Referential Integrity Failure: NDVI records exist for field_ids that do not exist "
            f"in the GeoJSON boundaries file: {sorted(list(unmatched_csv_ids))}"
        )
        report["success"] = False
        
    # 2. Detect fields that have geometry but no NDVI observations (warning, not error)
    unmatched_geojson_ids = geojson_ids - csv_ids
    if unmatched_geojson_ids:
        report["warnings"].append(
            f"Fields defined in GeoJSON have no corresponding NDVI time-series records: "
            f"{sorted(list(unmatched_geojson_ids))}"
        )
        
    return report

def load_and_validate_all(geojson_path: str = None, csv_path: str = None) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, Any]]:
    """Convenience function that loads and runs all checks.

    Args:
        geojson_path: Custom path to GeoJSON file (optional).
        csv_path: Custom path to CSV file (optional).

    Returns:
        A tuple of (GeoDataFrame, DataFrame, validation_summary_dict).
    """
    summary = {
        "success": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        gdf = load_field_boundaries(geojson_path)
        df = load_ndvi_data(csv_path)
    except Exception as e:
        summary["success"] = False
        summary["errors"].append(f"Ingestion failed: {e}")
        return gpd.GeoDataFrame(), pd.DataFrame(), summary
        
    # Run validators
    gdf_report = validate_field_data(gdf)
    df_report = validate_ndvi_data(df)
    ids_report = validate_field_ids(gdf, df)
    
    # Consolidate reports
    summary["success"] = gdf_report["success"] and df_report["success"] and ids_report["success"]
    summary["errors"].extend(gdf_report["errors"] + df_report["errors"] + ids_report["errors"])
    summary["warnings"].extend(gdf_report["warnings"] + df_report["warnings"] + ids_report["warnings"])
    
    return gdf, df, summary
