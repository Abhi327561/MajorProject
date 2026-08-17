# AI-Based Intelligent Cropland Monitoring & Crop Health Analysis

This is a final-year college project repository. The primary focus of this specific branch is the implementation of the **Crop Health Dashboard** module.

## Project Modules
- **CNN-Transformer field segmentation** (Teammate module)
- **Sentinel-2 satellite imagery processing** (Teammate module)
- **NDVI computation** (Teammate module)
- **Crop Health Dashboard** (This module)

---

## Crop Health Dashboard Module

The Crop Health Dashboard is built using **Streamlit**, **Pandas**, **GeoPandas**, **Plotly**, **Folium**, and **ReportLab**. It visualizes temporal vegetation indices (NDVI), identifies anomalies (crop stress, pest attacks, droughts), displays interactive field maps colored by current crop health, and generates professional PDF reports for farmers.

### Project Architecture & Directory Layout

The workspace is organized to keep teammate integrations modular and separate. All code related to the Crop Health Dashboard resides within the `crop_health_dashboard/` directory.

```
MajorProject/
│
├── crop_health_dashboard/
│   ├── app.py                      # Main Streamlit application entry point (Phase 4)
│   ├── config.py                   # Centralized configuration (thresholds, paths, map styles)
│   ├── data/
│   │   ├── mock_fields.geojson     # Generated mock field boundary data
│   │   └── mock_ndvi_timeseries.csv# Generated mock NDVI time-series data
│   │
│   └── src/
│       ├── __init__.py
│       ├── data_loader.py          # GeoJSON and CSV parser (Phase 2)
│       ├── ndvi_analysis.py        # Anomaly detection and statistics (Phase 2)
│       ├── map_utils.py            # Folium map generation and styling (Phase 3)
│       ├── report_generator.py     # PDF report generation (Phase 5)
│       └── mock_generator.py       # Standalone script to generate mock dataset files
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Phase 1: Setup and Mock Data Validation

In Phase 1, we set up the directory structure, configured the system parameters, and generated a realistic dataset representing fictional agricultural fields with various crop cycles and stress scenarios.

### Configured Thresholds (`crop_health_dashboard/config.py`)
- **Healthy Crop Threshold:** NDVI $\ge 0.60$
- **Poor Crop/Soil Threshold:** NDVI $< 0.30$
- **Moderate Crop Range:** $0.30 \le \text{NDVI} < 0.60$
- **Anomaly Drop Threshold:** $\ge 0.20$ drop between consecutive monthly acquisitions.
- **Maximum Cloud Cover Filter:** $20.0\%$ (Acquisitions with cloud cover greater than this are filtered from crop health metrics).

### Mock Fields Scenarios
1. **F01 (North Wheat Field):** Healthy and stable double-crop cycle (high NDVI).
2. **F02 (West Ridge Field):** Poor/fallow field with consistently low NDVI ($0.15 - 0.28$).
3. **F03 (East Maize Field):** Growing crop cycle starting low and rising to healthy peaks, dropping post-harvest in winter.
4. **F04 (South Barley Field):** Stress event scenario. Healthy early-season NDVI ($0.72$), followed by a sudden insect/drought drop in July to $0.38$ (a drop of $0.34$), and gradual recovery.
5. **F05 (Central Clover Field):** Moderate health ($0.35 - 0.58$) with normal variation.

---

## Installation & Running Validation

To set up the environment and run the internal consistency checks:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Mock Data Programmatically (Optional - Already generated):**
   ```bash
   python crop_health_dashboard/src/mock_generator.py
   ```

3. **Validate Dataset Quality & Consistency:**
   This command executes a standard-library check to verify matching geometries, schemas, referential integrity between GeoJSON/CSV, and check that the mock data contains the necessary scenarios (healthy, moderate, poor, cloud cover, and anomaly drops).
   ```bash
   python crop_health_dashboard/src/validate_data.py
   ```
