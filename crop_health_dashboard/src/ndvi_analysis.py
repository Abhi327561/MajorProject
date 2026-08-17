import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
    NDVI_POOR_THRESHOLD = config.NDVI_POOR_THRESHOLD
    NDVI_HEALTHY_THRESHOLD = config.NDVI_HEALTHY_THRESHOLD
    ANOMALY_DROP_THRESHOLD = config.ANOMALY_DROP_THRESHOLD
    MAX_CLOUD_COVER = config.MAX_CLOUD_COVER
except ImportError:
    # Fallback default values
    NDVI_POOR_THRESHOLD = 0.30
    NDVI_HEALTHY_THRESHOLD = 0.60
    ANOMALY_DROP_THRESHOLD = 0.20
    MAX_CLOUD_COVER = 20.0

def classify_health(ndvi: float, poor_thresh: float = NDVI_POOR_THRESHOLD, healthy_thresh: float = NDVI_HEALTHY_THRESHOLD) -> str:
    """Classifies NDVI value into Poor, Moderate, or Healthy.

    Args:
        ndvi: The NDVI value to classify.
        poor_thresh: Upper bound for poor classification. Defaults to config value.
        healthy_thresh: Lower bound for healthy classification. Defaults to config value.

    Returns:
        A string classification: 'Poor', 'Moderate', or 'Healthy'. Returns 'Unknown' if ndvi is NaN.
    """
    if ndvi is None or pd.isna(ndvi):
        return "Unknown"
    
    if ndvi < poor_thresh:
        return "Poor"
    elif ndvi <= healthy_thresh:
        return "Moderate"
    else:
        return "Healthy"

def process_field_timeseries(
    df_field: pd.DataFrame, 
    poor_thresh: float = NDVI_POOR_THRESHOLD, 
    healthy_thresh: float = NDVI_HEALTHY_THRESHOLD,
    drop_thresh: float = ANOMALY_DROP_THRESHOLD,
    max_cloud: float = MAX_CLOUD_COVER
) -> pd.DataFrame:
    """Processes time-series data for a single field.

    Applies cloud cover filtering, health classification, delta calculations,
    and anomaly detection.

    Args:
        df_field: DataFrame containing NDVI records for a single field_id.
        poor_thresh: Poor threshold.
        healthy_thresh: Healthy threshold.
        drop_thresh: Threshold for flagging a sudden drop anomaly.
        max_cloud: Maximum cloud cover percentage allowed for valid readings.

    Returns:
        A new DataFrame with analysis columns appended.
    """
    if df_field.empty:
        return df_field.copy()
        
    # Sort chronological
    df_sorted = df_field.sort_values(by="acquisition_date").copy()
    
    # 1. Cloud-cover filtering classification
    # Distinguish: 'Valid' (cloud <= max_cloud), 'Cloudy' (cloud > max_cloud), 'Excluded' (if missing/null)
    df_sorted['data_status'] = 'Valid'
    df_sorted.loc[df_sorted['cloud_cover'] > max_cloud, 'data_status'] = 'Cloudy'
    df_sorted.loc[df_sorted['ndvi_mean'].isna(), 'data_status'] = 'Excluded'
    
    # 2. Extract NDVI for valid observations only, to perform forward delta comparison
    # We create a helper column for 'valid' NDVI, which propagates forward, so we ignore cloudy observations when comparing deltas
    df_sorted['valid_ndvi_mean'] = df_sorted['ndvi_mean']
    df_sorted.loc[df_sorted['data_status'] != 'Valid', 'valid_ndvi_mean'] = np.nan
    
    # Forward fill valid values to have previous valid NDVI available for comparison
    df_sorted['prev_valid_ndvi'] = df_sorted['valid_ndvi_mean'].shift(1).ffill()
    
    # 3. Deltas and percentage changes (only computed for currently valid observations compared to previous valid observations)
    df_sorted['ndvi_delta'] = np.nan
    df_sorted['ndvi_pct_change'] = np.nan
    
    valid_mask = df_sorted['data_status'] == 'Valid'
    
    # Compute delta: current_ndvi - previous_valid_ndvi
    df_sorted.loc[valid_mask, 'ndvi_delta'] = df_sorted.loc[valid_mask, 'ndvi_mean'] - df_sorted.loc[valid_mask, 'prev_valid_ndvi']
    
    # Compute pct change: (delta / prev_ndvi) * 100 (avoid division-by-zero)
    # We safely divide by prev_valid_ndvi where it is non-zero
    prev_not_zero = (df_sorted['prev_valid_ndvi'] != 0) & df_sorted['prev_valid_ndvi'].notna()
    pct_mask = valid_mask & prev_not_zero
    df_sorted.loc[pct_mask, 'ndvi_pct_change'] = (df_sorted.loc[pct_mask, 'ndvi_delta'] / df_sorted.loc[pct_mask, 'prev_valid_ndvi']) * 100.0
    
    # 4. Anomaly detection: flag when ndvi_delta <= -drop_thresh (i.e. drop is >= drop_thresh)
    df_sorted['is_anomaly'] = False
    df_sorted['anomaly_severity'] = 'None'
    df_sorted['ndvi_drop'] = np.nan
    
    # An anomaly drop is a negative delta
    anomaly_mask = valid_mask & (df_sorted['ndvi_delta'] <= -drop_thresh)
    df_sorted.loc[anomaly_mask, 'is_anomaly'] = True
    df_sorted.loc[anomaly_mask, 'ndvi_drop'] = -df_sorted.loc[anomaly_mask, 'ndvi_delta']
    
    # Categorize severity
    # Drop between drop_thresh and 0.35: Moderate
    # Drop >= 0.35: Severe
    mod_severity_mask = anomaly_mask & (-df_sorted['ndvi_delta'] < 0.35)
    sev_severity_mask = anomaly_mask & (-df_sorted['ndvi_delta'] >= 0.35)
    df_sorted.loc[mod_severity_mask, 'anomaly_severity'] = 'Moderate'
    df_sorted.loc[sev_severity_mask, 'anomaly_severity'] = 'Severe'
    
    # 5. Crop health classification
    df_sorted['health_status'] = df_sorted['ndvi_mean'].apply(
        lambda x: classify_health(x, poor_thresh, healthy_thresh)
    )
    
    # Clean up intermediate columns we don't need in final output
    df_sorted.drop(columns=['valid_ndvi_mean'], inplace=True)
    
    return df_sorted

def process_all_fields(
    df: pd.DataFrame,
    poor_thresh: float = NDVI_POOR_THRESHOLD, 
    healthy_thresh: float = NDVI_HEALTHY_THRESHOLD,
    drop_thresh: float = ANOMALY_DROP_THRESHOLD,
    max_cloud: float = MAX_CLOUD_COVER
) -> pd.DataFrame:
    """Processes time-series data for all fields in the DataFrame.

    Args:
        df: Input DataFrame containing multi-field observations.
        poor_thresh: Poor threshold.
        healthy_thresh: Healthy threshold.
        drop_thresh: Anomaly drop threshold.
        max_cloud: Max cloud cover.

    Returns:
        A consolidated DataFrame containing processed metrics for all fields.
    """
    processed_dfs = []
    for field_id, group in df.groupby("field_id"):
        processed_field = process_field_timeseries(
            group, poor_thresh, healthy_thresh, drop_thresh, max_cloud
        )
        processed_dfs.append(processed_field)
        
    if not processed_dfs:
        return pd.DataFrame()
        
    return pd.concat(processed_dfs, ignore_index=True)

def detect_missing_months(df: pd.DataFrame, year: int = 2026) -> Dict[str, List[int]]:
    """Identifies months for which a field has no valid NDVI observation.

    Args:
        df: Input NDVI DataFrame (can be raw or processed).
        year: The year to check missing months for. Defaults to 2026.

    Returns:
        A dictionary mapping field_ids to a list of missing month integers (1-12).
    """
    missing_report = {}
    
    # Make sure we have datetime objects
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['acquisition_date']):
        df['acquisition_date'] = pd.to_datetime(df['acquisition_date'])
        
    # Standard 12 months check
    all_months = set(range(1, 13))
    
    for field_id, group in df.groupby("field_id"):
        # Filter for the target year
        year_mask = group['acquisition_date'].dt.year == year
        
        # Consider ONLY "Valid" observations (cloud_cover <= MAX_CLOUD_COVER and ndvi is not null)
        # Note: If it's the raw dataframe we filter by cloud_cover and nulls.
        # If it's processed, we filter where data_status == 'Valid'.
        if 'data_status' in group.columns:
            valid_observations = group[group['data_status'] == 'Valid']
        else:
            valid_observations = group[(group['cloud_cover'] <= MAX_CLOUD_COVER) & (group['ndvi_mean'].notna())]
            
        existing_months = set(valid_observations['acquisition_date'].dt.month.unique())
        missing_months = sorted(list(all_months - existing_months))
        missing_report[field_id] = missing_months
        
    return missing_report

def generate_field_summaries(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Generates field-level summary statistics based on processed time-series.

    Args:
        df_processed: Processed DataFrame containing delta, health, and anomaly flags.

    Returns:
        A DataFrame with one row per field, summarizing its latest telemetry.
    """
    summaries = []
    
    for field_id, group in df_processed.groupby("field_id"):
        # Sort chronologically to make sure we extract the latest record
        group_sorted = group.sort_values(by="acquisition_date")
        
        # 1. Total count of valid observations
        valid_obs = group_sorted[group_sorted['data_status'] == 'Valid']
        total_valid = len(valid_obs)
        
        # 2. Total count of anomalies
        total_anomalies = group_sorted['is_anomaly'].sum()
        
        # 3. Extract the latest observation (even if cloudy, to display latest data state)
        latest_record = group_sorted.iloc[-1]
        
        # 4. Extract the latest VALID observation for representing current health state
        latest_valid_record = valid_obs.iloc[-1] if not valid_obs.empty else None
        
        # Summary dict
        summary = {
            "field_id": field_id,
            "latest_acquisition_date": latest_record['acquisition_date'],
            "latest_ndvi_mean": latest_record['ndvi_mean'],
            "latest_cloud_cover": latest_record['cloud_cover'],
            "latest_data_status": latest_record['data_status'],
            
            # Health is based on latest valid observation if possible, fallback to raw latest
            "latest_health_status": latest_valid_record['health_status'] if latest_valid_record is not None else latest_record['health_status'],
            
            # Anomaly details
            "latest_is_anomaly": latest_record['is_anomaly'],
            "latest_anomaly_severity": latest_record['anomaly_severity'],
            
            # Historical summary
            "total_anomalies": int(total_anomalies),
            "total_valid_observations": int(total_valid)
        }
        
        # Previous valid NDVI and delta (from the latest valid observation's point of view)
        if latest_valid_record is not None:
            summary["latest_valid_ndvi_mean"] = latest_valid_record['ndvi_mean']
            summary["prev_valid_ndvi_mean"] = latest_valid_record['prev_valid_ndvi']
            summary["latest_ndvi_delta"] = latest_valid_record['ndvi_delta']
            summary["latest_ndvi_pct_change"] = latest_valid_record['ndvi_pct_change']
        else:
            summary["latest_valid_ndvi_mean"] = np.nan
            summary["prev_valid_ndvi_mean"] = np.nan
            summary["latest_ndvi_delta"] = np.nan
            summary["latest_ndvi_pct_change"] = np.nan
            
        summaries.append(summary)
        
    return pd.DataFrame(summaries)
