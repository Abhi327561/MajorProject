# AI-Based Intelligent Cropland Monitoring & Crop Health Analysis

This is the final-year college project repository. The primary focus of this module is the **Crop Health Dashboard** implementation.

---

## 1. Crop Health Dashboard Overview
The **Crop Health Dashboard** is a farmer-facing interface designed to monitor vegetation greenness index trajectories over time, detect sudden anomalies/stress events (due to drought, weed competition, pest attacks, or disease), visualize field boundaries, and compile downloadable PDF health summaries.

---

## 2. Directory Layout & Architecture
```
MajorProject/
│
├── crop_health_dashboard/
│   ├── app.py                      # Streamlit dashboard interface
│   ├── config.py                   # Thresholds, colors, paths, mapping aliases
│   ├── data/
│   │   ├── mock_fields.geojson     # Polygon boundaries for agricultural fields
│   │   └── mock_ndvi_timeseries.csv# NDVI observations with cloud cover
│   │
│   └── src/
│       ├── __init__.py
│       ├── data_loader.py          # Data ingestion, column mapping & bounds validation
│       ├── ndvi_analysis.py        # Analytics (deltas, classification, anomalies)
│       ├── report_generator.py     # ReportLab PDF report compiler
│       ├── mock_generator.py       # Standalone generator for mock datasets
│       └── test_processing.py      # 14-test regression and unit test suite
│
├── requirements.txt                # System requirements
└── README.md                       # This file
```

The data flow is structured as follows:
```
REAL TEAM DATA / MOCK DATA
          ↓
  data_loader.py (Normalization & Schema Validation)
          ↓
  ndvi_analysis.py (Analytics & Anomaly Core)
          ↓
  app.py (Streamlit Visualization & Map Controls)
          ↓
  report_generator.py (PDF Report Compiler)
```

---

## 3. Configured Parameter Thresholds (`config.py`)
- **Healthy Crop Threshold:** NDVI $\ge 0.60$ (Shaded green band)
- **Moderate Crop Health:** $0.30 \le \text{NDVI} < 0.60$ (Shaded orange/amber band)
- **Poor Crop Health:** NDVI $< 0.30$ (Shaded crimson/red band)
- **Stress Drop Anomaly Threshold:** $\ge 0.20$ drop between consecutive valid acquisitions.
- **Maximum Cloud Filter Threshold:** $20.0\%$ (Acquisitions with cloud cover exceeding this are excluded from status assessments by default).

---

## 4. Key Functional Features
1. **Interactive Field Map (Folium):** Centers automatically on the selected field boundary. The boundary polygon is color-coded in real-time according to the selected date's health status (Green: Healthy, Yellow/Orange: Moderate, Red: Poor, Grey: Cloudy/Excluded).
2. **Interactive Plotly Time-Series:** Visualizes the crop trajectory across all dates. Includes color-shaded horizontal health zones, highlighting cloudy observations and stress events with markers, and dynamically updating with sidebar controls.
3. **Farmer PDF Report Generator:** Compiles a layout containing metadata, current parameters, relative change, stress analysis, and actionable advice.
4. **Data Normalization Layer:** Integrates columns from teammates' models. Renaming configurations can be modified directly within `crop_health_dashboard/config.py` (e.g., mapping custom column names to internal schemas).
5. **Robust Validation:** Checks referential integrity, ranges, date formats, nulls, duplicates, and boundary voids.

---

## 5. Installation & Usage Instructions

### Step 1: Initialize Virtual Environment & Install Dependencies
Run the following commands inside PowerShell or CMD to set up:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Run Consistency Validation Checks
Validate that the loaded GeoJSON and NDVI CSV are consistent:
```powershell
python crop_health_dashboard/src/test_processing.py
```
This runs the 14 automated unit/regression tests covering schema mappings, bounds checks, anomalies, and PDF compilation.

### Step 3: Run the Streamlit Dashboard
```powershell
python -m streamlit run crop_health_dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 6. Real-Data Integration Support
To load real data from teammate modules (field segmentation, Sentinel-2 preprocessing, or NDVI computation):
1. Place their `.geojson` and `.csv` files under the `crop_health_dashboard/data/` folder.
2. Open `crop_health_dashboard/config.py`.
3. Update `GEOJSON_PATH` and `CSV_PATH` to point to their files.
4. Update `COLUMN_MAPPING_GEOJSON` and `COLUMN_MAPPING_NDVI` dictionaries to map their output columns to the internal variables required by the dashboard.
