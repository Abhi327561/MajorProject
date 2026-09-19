import os

# Centralized configuration parameters for the Crop Health Dashboard

# Thresholds for Crop Health Classification
# Poor health: NDVI < NDVI_POOR_THRESHOLD
# Moderate health: NDVI_POOR_THRESHOLD <= NDVI < NDVI_HEALTHY_THRESHOLD
# Healthy: NDVI >= NDVI_HEALTHY_THRESHOLD
NDVI_POOR_THRESHOLD = 0.30
NDVI_HEALTHY_THRESHOLD = 0.60

# Threshold for detecting anomalies (stress events)
# An anomaly is flagged if there is a drop in NDVI of at least this value compared to the previous month
ANOMALY_DROP_THRESHOLD = 0.20

# Data Filtering Configuration
# Satellite acquisitions with cloud cover percentage above this threshold will be filtered out
MAX_CLOUD_COVER = 20.0  # percentage

# Paths to Data Files (Model outputs and Benchmark datasets)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "mock_fields.geojson")
CSV_PATH = os.path.join(DATA_DIR, "mock_ndvi_timeseries.csv")

# Real Model Output Paths
MODEL_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "outputs", "ndvi")
MODEL_CSV_PATH = os.path.join(MODEL_OUTPUTS_DIR, "field_ndvi.csv")
MODEL_HEALTH_PATH = os.path.join(MODEL_OUTPUTS_DIR, "field_health_classification.csv")
MODEL_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "outputs", "predictions_test.geojson")

# Column Normalization Mapping for teammate / Model integration
COLUMN_MAPPING_GEOJSON = {
    "field_id": "field_id",
    "field_name": "field_name",
    "area_ha": "area_ha",
    "geometry": "geometry"
}

COLUMN_MAPPING_NDVI = {
    "field_id": "field_id",
    "acquisition_date": "acquisition_date",
    "date": "acquisition_date",
    "ndvi_mean": "ndvi_mean",
    "mean_ndvi": "ndvi_mean",
    "ndvi_median": "ndvi_median",
    "median_ndvi": "ndvi_median",
    "cloud_cover": "cloud_cover"
}


# Map Configuration
MAP_DEFAULT_ZOOM = 15
MAP_CENTER_LATITUDE = 29.7200  # Agricultural cropland basin (Karnal farm belt)
MAP_CENTER_LONGITUDE = 76.9500

# UI Styling Hex Colors
COLOR_HEALTHY = "#2CA02C"   # Deep green
COLOR_MODERATE = "#FF7F0E"  # Orange/Amber
COLOR_POOR = "#D62728"      # Crimson/Red
COLOR_ANOMALY = "#9467BD"   # Muted purple for anomaly flags
