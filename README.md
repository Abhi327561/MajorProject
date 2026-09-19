# AI-Based Intelligent Cropland Monitoring & Crop Health Analysis

An end-to-end intelligent agricultural monitoring system combining deep learning-based field boundary segmentation (**CTHBNet**) with an interactive temporal crop health analysis and reporting dashboard.

---

## 📌 System Architecture & Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Sentinel-2 Multi-Spectral Imagery                      │
│                  (B02 Blue, B03 Green, B04 Red, B08 NIR)                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Module 1: Deep Learning Segmentation (model/)                │
│  - CTHBNet Architecture (Hybrid CNN + Vision Transformer Encoder-Decoder)   │
│  - Boundary-Aware Loss (BCE + Dice + Sobel-based Edge Loss)                 │
│  - Multi-Scale Gated Fusion (Local Edge Detail + Global Spatial Context)    │
│  - Metrics: Boundary F1, Region IoU, Precision, Recall                      │
│  - Checkpoint Progression Evaluation Across 30 Epochs                       │
│  - Zonal Multi-Temporal NDVI Extraction & Health Classification Engine      │
│  - Automated GeoJSON Polygon Boundary Extraction                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                     Model Extractions (GeoJSON & NDVI CSV)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          Module 2: Crop Health & Analytics Dashboard (crop_health_dashboard/)│
│  - Data Ingestion, Schema Normalization & Dynamic Multi-Source Switching    │
│  - NDVI Trajectory Analytics & Stress Drop Anomaly Detection                │
│  - Interactive Geospatial Mapping (Folium) with Dynamic Status Coloring     │
│  - High-Resolution Multi-Spectral Satellite Canopy Thumbnail Visualizer     │
│  - Time-Series Trajectory Visualizer (Plotly) with Cloud Masking            │
│  - Automated Farmer PDF Report Compiler (ReportLab)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
MajorProject/
│
├── model/                          # Deep Learning Segmentation Pipeline (CTHBNet)
│   ├── architecture.py             # CTHBNet model (CNN branch + ViT branch + Gated Fusion + Decoder)
│   ├── losses.py                   # Boundary-Aware Loss (BCE + Dice + Sobel boundary loss)
│   ├── metrics.py                  # Evaluation metrics (IoU, Boundary F1, Precision, Recall)
│   ├── dataset.py                  # AI4Boundaries Sentinel-2 NetCDF + GeoTIFF Dataset Loader
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
│   ├── checkpoints/                # 30 Saved epoch checkpoints (best.pt, last.pt, epoch_001.pt...)
│   └── outputs/                    # Overlay figures, exported GeoJSONs, and NDVI extractions
│       ├── overlays/               # Prediction vs GT boundary segmentation overlay images
│       └── ndvi/                   # 25,000+ extracted temporal NDVI records & classifications
│
├── crop_health_dashboard/          # Interactive Streamlit Web Application
│   ├── app.py                      # Main Streamlit dashboard UI & visual controls
│   ├── config.py                   # Classification thresholds, colors, paths & column aliases
│   ├── data/
│   │   ├── mock_fields.geojson     # Benchmark GeoJSON polygon boundaries for cropland fields
│   │   ├── mock_ndvi_timeseries.csv# NDVI observations with cloud cover & date indices
│   │   └── imagery/                # 60 Sentinel-2 false-color NDVI crop canopy thumbnails
│   └── src/
│       ├── __init__.py
│       ├── data_loader.py          # Data ingestion, schema validation & bounds checking
│       ├── ndvi_analysis.py        # Analytics core (NDVI classification, delta & anomaly engine)
│       ├── report_generator.py     # PDF health report compiler using ReportLab
│       ├── generate_imagery.py     # Multi-spectral thumbnail generator
│       ├── mock_generator.py       # Benchmark synthetic scenario data generator
│       ├── validate_data.py        # Dataset consistency & schema validator
│       └── test_processing.py      # Automated 14-test regression and validation suite
│
├── requirements.txt                # Root dependencies (Streamlit, Folium, Plotly, GeoPandas, etc.)
└── README.md                       # Comprehensive project documentation
```

---

## 🧠 Module 1: CTHBNet Field Boundary Segmentation (`model/`)

`CTHBNet` (**C**onvolution-**T**ransformer **H**ybrid **B**oundary **N**etwork) is a specialized deep learning architecture designed specifically for extracting agricultural parcel boundaries from Sentinel-2 satellite imagery chips.

### 📐 1. Architectural Architecture

```
Input Image Chip (4, 256, 256) [B02 Blue, B03 Green, B04 Red, B08 NIR]
  │
  ├──► [CNN Encoder] ── (4-Stage ResNet/U-Net Blocks) ──► Low-level edge & texture features
  │        │ (Skip connections: s1, s2, s3, s4)
  │        ▼
  │   Bottleneck Features (256, 16, 16)
  │        │
  │        ├──► [Transformer Encoder] (ViT Patch Embedding + 4 MHSA Blocks) ──► Global spatial geometry
  │        │
  │        ▼
  │   [Gated Fusion Module] (Learnable per-pixel gate: α * F_CNN + (1 - α) * F_ViT)
  │        │
  │        ▼
  └──► [Decoder with Skip Connections] (Multi-scale upsampling & concatenation)
           │
           ▼
     Output Boundary Logits (1, 256, 256) ──► Sigmoid ──► Predicted Boundary Mask
```

#### Key Components:
1. **CNN Branch:** 4-stage convolutional encoder capturing fine-grained local textures, spectral contrast gradients, and precise local edge positions. Preserves multi-scale skip connections ($s_1, s_2, s_3, s_4$) for the decoder.
2. **Transformer Branch:** Operates on the deepest bottleneck feature map ($16 \times 16$). Patchifies features into a sequence and applies Multi-Head Self-Attention (4 Transformer layers, 8 heads, 256 embedding dimension) to capture long-range global parcel geometry, field regularities, and relationships between adjacent crop parcels.
3. **Gated Fusion Module:** Combines CNN local features ($F_{CNN}$) and Transformer global representations ($F_{ViT}$) using a learnable gating parameter $\alpha \in [0, 1]$:
   $$F_{Fused} = \sigma(W_g [F_{CNN}, F_{ViT}]) \odot F_{CNN} + (1 - \sigma(W_g [F_{CNN}, F_{ViT}])) \odot F_{ViT}$$
4. **Decoder:** Progressively upsamples fused representations back to full $256 \times 256$ resolution using skip connections from the CNN encoder to produce sharp, closed boundary polygons.

---

### 📉 2. Boundary-Aware Loss Function

Because agricultural field boundaries constitute a very small percentage of total image pixels compared to field interiors, standard Binary Cross-Entropy (BCE) causes blurred or offset boundaries. CTHBNet uses a composite **Boundary-Aware Loss**:

$$\mathcal{L}_{Total} = w_{bce} \mathcal{L}_{BCE} + w_{dice} \mathcal{L}_{Dice} + w_{boundary} \mathcal{L}_{Boundary}$$

- **$\mathcal{L}_{BCE}$:** Standard pixel-wise binary cross entropy.
- **$\mathcal{L}_{Dice}$:** Soft-Dice loss ensuring region-level overlap invariance to class imbalance:
  $$\mathcal{L}_{Dice} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}$$
- **$\mathcal{L}_{Boundary}$:** Edge-focused loss where a 2D Sobel filter extracts ground-truth boundary bands and up-weights BCE errors occurring directly on boundary pixels.

---

### 📊 3. Evaluation Metrics

| Metric | Formulation / Definition | Description |
|---|---|---|
| **IoU (Jaccard Index)** | $\frac{|P \cap G|}{|P \cup G|}$ | Overall region segmentation overlap quality |
| **Precision / Recall / F1** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Standard pixel classification quality |
| **Boundary F1 ($\text{BF1}$)** | Precision/Recall within tolerance radius $r=2\text{px}$ | Measures boundary edge sharpness and localization accuracy |

---

### 📈 4. Training Pipeline & Progression

- **Optimizer:** AdamW ($\text{LR} = 10^{-4}$, $\text{Weight Decay} = 10^{-4}$)
- **LR Schedule:** Cosine Annealing with 2-epoch Warmup
- **Checkpoints Saved:** `checkpoints/best.pt`, `checkpoints/last.pt`, and snapshots across all 30 epochs (`epoch_001.pt` to `epoch_030.pt`).
- **Checkpoint Progression Tracking:** Standalone script `evaluate_progression.py` evaluates all 30 epoch checkpoints sequentially and outputs metrics evolution to CSV and plots.

---

### 🛰️ 5. Zonal Multi-Temporal NDVI Extraction (`ndvi_analysis.py`)

Takes multi-temporal Sentinel-2 NetCDF files and overlay masks to compute zonal temporal crop health statistics:
- Analyzed 43 Sentinel-2 observation chips across 6 monthly time-steps.
- Extracted **25,040+ temporal records** across **4,173 individual agricultural parcels**.
- Computed `median_ndvi`, `mean_ndvi`, and `pixel_count` for each field across all dates.
- Classified overall health trajectories into **Healthy**, **Moderate**, and **Poor** in `field_health_classification.csv`.

---

### 🖼️ 6. Visual Overlay Extraction (`visualize.py`)

Visual overlay generation script produces three-channel composite images under `model/outputs/overlays/`:
- <span style="color:#00ffff;font-weight:bold;">■ Cyan:</span> Ground Truth field boundary
- <span style="color:#ff00ff;font-weight:bold;">■ Magenta:</span> CTHBNet Model Predicted boundary
- <span style="color:#ffff00;font-weight:bold;">■ Yellow:</span> Agreement / Overlap region

---

## 🌾 Module 2: Crop Health & Analytics Dashboard (`crop_health_dashboard/`)

An interactive decision-support interface connecting model extractions with farmer decision-making:
1. **Interactive Spatial View (Folium):** Rendered on satellite basemaps with dynamic health color badges (Green $\ge 0.60$, Orange $0.30 - 0.60$, Red $< 0.30$).
2. **Sentinel-2 Multi-Spectral Thumbnail Panel:** Visualizes false-color canopy heatmaps with crop boundary overlays.
3. **Temporal NDVI Trajectory (Plotly):** Multi-month time-series curves with cloud cover filters and stress drop anomaly flags.
4. **Automated Farmer PDF Report (ReportLab):** Generates downloadable field health advisory reports with customized agronomic recommendations.

---

## 🚀 Execution Guide & Commands

### 1. Environment Setup
```powershell
# Create & activate environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r model/requirements.txt
```

### 2. Model Training & Evaluation
```powershell
cd model

# Run single batch overfit sanity test
python overfit_test.py

# Train CTHBNet across 30 epochs
python train.py --epochs 30 --lr 1e-4 --batch_size 4

# Evaluate best checkpoint on test set
python evaluate.py --checkpoint checkpoints/best.pt --split test

# Evaluate progression across all 30 epoch checkpoints
python evaluate_progression.py --split val --out_csv outputs/progression.csv

# Generate predicted vs GT boundary overlays
python visualize.py --checkpoint checkpoints/best.pt --split test --num_samples 6

# Extract multi-temporal zonal NDVI statistics
python ndvi_analysis.py

# Export predicted boundaries as GeoJSON
python export_geojson.py --checkpoint checkpoints/best.pt --split test --out outputs/predictions.geojson
```

### 3. Launch the Crop Health Dashboard
```powershell
cd ..
# Run regression tests
python crop_health_dashboard/src/test_processing.py

# Launch Streamlit application
python -m streamlit run crop_health_dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.
