# AI-Based Intelligent Cropland Monitoring & Crop Health Analysis

An end-to-end intelligent agricultural monitoring system combining deep learning-based field boundary segmentation (CTHBNet) with an interactive temporal crop health analysis and reporting dashboard.

---

## 📌 System Architecture & Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Sentinel-2 Multi-Spectral Imagery                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Module 1: Deep Learning Segmentation (model/)                │
│  - CTHBNet Architecture (Hybrid CNN + Vision Transformer Encoder-Decoder)   │
│  - Boundary-Aware Loss (BCE + Dice + Sobel-based Edge Loss)                 │
│  - Metrics: Boundary F1, IoU, Precision, Recall                             │
│  - Checkpoint Progression Evaluation & Visual Overlays                      │
│  - Automated GeoJSON Polygon Boundary Extraction                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                       GeoJSON Fields & NDVI Timeseries
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            Module 2: Crop Health & Analytics Dashboard (dashboard/)          │
│  - Data Ingestion, Schema Normalization & Bounds Validation                 │
│  - NDVI Trajectory Analytics & Stress Drop Anomaly Detection                │
│  - Interactive Geospatial Mapping (Folium) with Dynamic Status Coloring     │
│  - Time-Series Trajectory Visualizer (Plotly) with Cloud Masking            │
│  - Automated Farmer PDF Report Compiler (ReportLab)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
MajorProject/
│
├── crop_health_dashboard/          # Interactive Streamlit Web Application
│   ├── app.py                      # Main Streamlit dashboard UI & visual controls
│   ├── config.py                   # Classification thresholds, colors, paths & column aliases
│   ├── data/
│   │   ├── mock_fields.geojson     # GeoJSON polygon boundaries for cropland fields
│   │   └── mock_ndvi_timeseries.csv# NDVI observations with cloud cover & date indices
│   └── src/
│       ├── __init__.py
│       ├── data_loader.py          # Data ingestion, schema validation & bounds checking
│       ├── ndvi_analysis.py        # Analytics core (NDVI classification, delta & anomaly engine)
│       ├── report_generator.py     # PDF health report compiler using ReportLab
│       ├── mock_generator.py       # Standalone generator for synthetic benchmark data
│       ├── validate_data.py        # Standard library dataset consistency & schema validator
│       └── test_processing.py      # Automated 14-test regression and validation suite
│
├── model/                          # Deep Learning Segmentation Pipeline (CTHBNet)
│   ├── architecture.py             # CTHBNet model (CNN branch + ViT branch + Gated Fusion + Decoder)
│   ├── losses.py                   # Boundary-Aware Loss (BCE + Dice + Sobel boundary loss)
│   ├── metrics.py                  # Evaluation metrics (IoU, Boundary F1, Precision, Recall)
│   ├── dataset.py                  # PyTorch Dataset wiring & MockCroplandDataset stand-in
│   ├── config.py                   # Model hyperparameters, optimizer settings & input shapes
│   ├── train.py                    # Training loop with AdamW, Cosine Annealing & checkpointing
│   ├── evaluate.py                 # Split-level model evaluation (Loss, IoU, Boundary F1)
│   ├── evaluate_progression.py     # Checkpoint-by-checkpoint validation progression tracking
│   ├── visualize.py                # Prediction vs. Ground-Truth overlay visualizer
│   ├── export_geojson.py           # Raster mask to georeferenced GeoJSON polygon exporter
│   ├── overfit_test.py             # Single-batch sanity & convergence verification test
│   ├── ndvi_analysis.py            # Sentinel-2 NetCDF extraction and zonal NDVI analysis
│   ├── plot_ndvi.py                # Plotting scripts for monthly/seasonal NDVI trends
│   ├── requirements.txt            # Deep learning dependencies (PyTorch, Torchvision, etc.)
│   ├── checkpoints/                # Saved model weights (last.pt, best.pt, epoch snapshots)
│   └── outputs/                    # Overlay figures, exported GeoJSONs, and NDVI plots
│
├── requirements.txt                # Root dependencies (Streamlit, Folium, Plotly, GeoPandas, etc.)
└── README.md                       # Project documentation
```

---

## 🧠 Module 1: CTHBNet Field Boundary Segmentation (`model/`)

`CTHBNet` (Convolution-Transformer Hybrid Boundary Network) is engineered specifically for delineating agricultural parcel boundaries from Sentinel-2 satellite imagery chips.

### Key Architectural Highlights
- **CNN Branch:** 4-stage U-Net style convolutional encoder capturing fine-grained local textures and low-level boundary gradients.
- **Transformer Branch:** Multi-Head Self-Attention (ViT encoder) operating on deep feature maps to learn global spatial context and parcel shape geometries.
- **Gated Fusion Module:** Learnable channel/spatial gating mechanism combining local boundary details with global context.
- **Decoder & Skip Connections:** Multi-scale feature upsampling to restore full resolution with sharp boundary localization.
- **Boundary-Aware Loss:** Combines Binary Cross-Entropy (BCE), Soft-Dice Loss, and a Sobel-filtered Edge Loss to prevent edge blurring and compensate for small boundary pixel fractions.

### Model Execution & Scripts
```bash
# Navigate to model folder
cd model

# 1. Sanity check: Run overfit test on a single batch
python overfit_test.py

# 2. Train CTHBNet (with AdamW, Cosine Annealing, and auto-checkpointing)
python train.py --epochs 30 --lr 1e-4 --batch_size 16

# 3. Resume training from a checkpoint
python train.py --resume checkpoints/last.pt --epochs 50

# 4. Evaluate performance on validation/test sets
python evaluate.py --checkpoint checkpoints/best.pt --split test

# 5. Track metric progression across all saved epoch checkpoints
python evaluate_progression.py --split val --out_csv outputs/metrics_progression.csv

# 6. Generate boundary visual overlay comparisons (cyan = GT, magenta = pred, yellow = overlap)
python visualize.py --checkpoint checkpoints/best.pt --split val --num_samples 6

# 7. Export predicted boundaries as GeoJSON polygons for the dashboard
python export_geojson.py --checkpoint checkpoints/best.pt --split test --out outputs/predictions.geojson
```

---

## 🌾 Module 2: Crop Health & Analytics Dashboard (`crop_health_dashboard/`)

An interactive, farmer-centric decision support interface providing temporal vegetation health insights and field analytics.

### Key Dashboard Capabilities
1. **Interactive Geospatial Map (Folium):** 
   - Dynamically zooms to selected field boundaries.
   - Color-codes field polygons based on selected date's health score (Green: Healthy, Yellow/Orange: Moderate, Red: Poor, Grey: Cloudy/Excluded).
2. **Temporal Vegetation Trajectories (Plotly):** 
   - Time-series plot of mean and median NDVI values.
   - Shaded health zone bands (Healthy: $\ge 0.60$, Moderate: $0.30 - 0.60$, Poor: $< 0.30$).
   - Marker annotations for cloud-contaminated passes and sudden stress drop events.
3. **Automated PDF Crop Health Reports (ReportLab):** 
   - One-click downloadable summary reports including farmer metadata, health status, stress alerts, and customized agronomic recommendations.
4. **Data Validation & Resilient Processing:**
   - Robust column mapping aliases in `config.py` to seamlessly connect external model outputs.
   - Built-in sanity checks for cloud cover masking, duplicate dates, and invalid coordinate geometries.

---

## ⚙️ Configuration & Parameter Thresholds

Configured in `crop_health_dashboard/config.py` and `model/config.py`:

| Parameter | Default Value | Description |
|---|---|---|
| `NDVI_HEALTHY_THRESHOLD` | `0.60` | NDVI values $\ge 0.60$ classified as **Healthy** |
| `NDVI_POOR_THRESHOLD` | `0.30` | NDVI values $< 0.30$ classified as **Poor / Severe Stress** |
| `ANOMALY_DROP_THRESHOLD` | `0.20` | Sudden NDVI drop $\ge 0.20$ between clear visits triggers stress alert |
| `MAX_CLOUD_COVER` | `20.0%` | Acquisitions exceeding cloud threshold are filtered from health scoring |
| `BOUNDARY_TOLERANCE_PX` | `2 px` | Tolerance radius for Boundary F1 evaluation |
| `IN_CHANNELS` | `4` | Input image channels (B02 Blue, B03 Green, B04 Red, B08 NIR) |
| `IMG_SIZE` | `256` | Chip dimension ($256 \times 256$ pixels) |

---

## 🚀 Installation & Quick Start

### 1. Environment Setup

```powershell
# Clone the repository and navigate to root directory
cd MajorProject

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dashboard and model dependencies
pip install -r requirements.txt
pip install -r model/requirements.txt
```

### 2. Run Test & Validation Suites

```powershell
# Run the 14-test regression and unit test suite
python crop_health_dashboard/src/test_processing.py

# Run standalone dataset consistency verification
python crop_health_dashboard/src/validate_data.py
```

### 3. Launch the Crop Health Dashboard

```powershell
# Start the Streamlit application
python -m streamlit run crop_health_dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 🔄 End-to-End Real Data Integration

To integrate outputs from upstream preprocessing or segmentation runs into the dashboard:
1. Export model polygon predictions using `model/export_geojson.py` or place your `.geojson` under `crop_health_dashboard/data/`.
2. Place your NDVI time-series CSV under `crop_health_dashboard/data/`.
3. Open `crop_health_dashboard/config.py` and update:
   - `GEOJSON_PATH` and `CSV_PATH`
   - `COLUMN_MAPPING_GEOJSON` and `COLUMN_MAPPING_NDVI` dictionaries to map your pipeline's column headers to the dashboard schema.
4. Refresh the Streamlit dashboard to visualize your updated data.
