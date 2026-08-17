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

# Paths to Data Files (can be replaced with real teammate files later)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GEOJSON_PATH = os.path.join(DATA_DIR, "mock_fields.geojson")
CSV_PATH = os.path.join(DATA_DIR, "mock_ndvi_timeseries.csv")

# Map Configuration
MAP_DEFAULT_ZOOM = 14
MAP_CENTER_LATITUDE = 28.6139  # Fictional cluster near Delhi coordinates (can be updated for any region)
MAP_CENTER_LONGITUDE = 77.2090

# UI Styling Hex Colors
COLOR_HEALTHY = "#2CA02C"   # Deep green
COLOR_MODERATE = "#FF7F0E"  # Orange/Amber
COLOR_POOR = "#D62728"      # Crimson/Red
COLOR_ANOMALY = "#9467BD"   # Muted purple for anomaly flags
