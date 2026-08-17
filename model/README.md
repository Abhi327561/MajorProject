# model/ — CTHBNet Field Boundary Segmentation

Model architecture, loss, training loop, evaluation, and export scripts for
**AI-Based Intelligent Cropland Monitoring & Crop Health Analysis**. This
component takes Sentinel-2 imagery chips and predicts per-pixel field
boundaries, then exports them as GeoJSON polygons for the NDVI dashboard.

## Contents

```
model/
├── architecture.py     # CTHBNet: CNN branch + Transformer branch + fusion + decoder
├── losses.py            # Dice + BCE + boundary-aware loss
├── metrics.py            # IoU, precision/recall/F1, boundary F1
├── dataset.py             # CroplandDataset import wiring + mock dataset stand-in
├── config.py               # all hyperparameters / paths in one place
├── train.py                 # training loop: AdamW, LR schedule, checkpoint + resume
├── visualize.py               # predicted-vs-GT boundary overlay images
├── export_geojson.py            # export predictions as GeoJSON polygons
├── requirements.txt
├── checkpoints/                  # saved checkpoints (see "Checkpoints" below)
└── outputs/                        # overlays, GeoJSON, contact sheets
```

## Setup

```bash
cd model
pip install -r requirements.txt
```

## Status: currently running against a mock dataset

`CroplandDataset` from the preprocessing pipeline isn't merged into this
repo yet, so `dataset.py` currently uses `MockCroplandDataset`, which
returns the same shape/format so the rest of the pipeline can be built and
tested independently:

```python
(image_tensor, mask_tensor, field_id)
# image_tensor: FloatTensor (C, H, W), default C=4, H=W=256
# mask_tensor:  FloatTensor (1, H, W), binary {0., 1.}
# field_id:     str
```

**To switch to the real dataset once merged**, in `dataset.py`:
1. Uncomment `from preprocessing.dataset import CroplandDataset`.
2. Fill in the real dataset construction in `get_datasets(use_mock=False, ...)`.
3. Run everything with `--no-mock` (train.py) or set `config.USE_MOCK_DATASET = False`.

If the real `CroplandDataset`'s input channel count or chip size differs
from `IN_CHANNELS=4` / `IMG_SIZE=256` in `config.py`, update those — the
architecture is fully shape-agnostic (it adapts to whatever `H, W` it's
given; only `in_channels` needs to match at construction time).

## Model architecture

`CTHBNet` (see `architecture.py`):
- **CNN branch**: 4-stage U-Net-style encoder, captures local edge/texture
  detail, keeps skip connections for the decoder.
- **Transformer branch**: patchifies the CNN's deepest feature map, runs it
  through a ViT-style encoder (multi-head self-attention) to capture
  global context — e.g. a field's overall shape and its relation to
  neighboring fields — then reshapes back to a spatial map.
- **Fusion module**: a learnable gate that combines local (CNN) and global
  (Transformer) features per-pixel.
- **Decoder**: U-Net-style upsampling back to full resolution using the
  CNN encoder's skip connections, producing a single-channel boundary logit
  map.

Input: `(B, C_in, H, W)`. Output: `(B, 1, H, W)` raw logits (apply
`sigmoid` for probabilities).

## Loss

`BoundaryAwareLoss` = `w_bce * BCE + w_dice * Dice + w_boundary * BoundaryLoss`.

The boundary term extracts a thin edge band from the ground-truth mask via
a Sobel filter and up-weights BCE loss on those pixels, since field
interiors dominate pixel count and a plain BCE/Dice loss under-penalizes
blurry or offset boundaries. Weights are configurable in `config.py`.

## Training

```bash
python train.py                              # fresh run, defaults from config.py
python train.py --epochs 30 --lr 1e-4 --batch_size 16
python train.py --no-mock                    # once CroplandDataset is wired in
```

- Optimizer: AdamW (`lr`, `weight_decay` configurable).
- LR scheduler: cosine annealing by default (`config.LR_SCHEDULER = "step"`
  for step decay instead).
- Gradient clipping enabled by default (`config.GRAD_CLIP_NORM`).

### Checkpointing & resuming (important for Colab)

Every epoch, `train.py` saves:
- `checkpoints/last.pt` — always overwritten, latest state. **This is what
  you resume from** after a Colab disconnect.
- `checkpoints/best.pt` — overwritten only when val boundary F1 improves.
- `checkpoints/epoch_XXX.pt` — periodic snapshot, controlled by
  `--ckpt_every N` (default every epoch; set `--ckpt_every 0` to disable).

Each checkpoint bundles model + optimizer + scheduler state, epoch number,
best-metric-so-far, and RNG state, so resuming continues exactly where
training left off:

```bash
python train.py --resume checkpoints/last.pt --epochs 50
```

If checkpoints get too large for git, push them to Drive and commit only a
text pointer (e.g. a `checkpoints/DRIVE_LINK.txt` with the shareable link)
instead of the `.pt` files.

## Evaluation metrics

Computed each epoch on the val split, and available standalone via
`metrics.compute_all_metrics(logits, target_mask)`:
- **IoU** — region-level intersection-over-union of the full predicted mask.
- **Precision / Recall / F1** — region-level pixel classification.
- **Boundary F1** — precision/recall/F1 computed only on the thin boundary
  band (Sobel-extracted), with a pixel tolerance radius
  (`config.BOUNDARY_TOLERANCE_PX`, default 2). This is the metric that
  actually reflects boundary sharpness — region IoU can look fine even
  with blurry or slightly offset edges.

## Visualizing predictions (for the review deck)

```bash
python visualize.py --checkpoint checkpoints/best.pt --split val --num_samples 6
```

Generates one overlay PNG per sample (cyan = ground truth boundary,
magenta = predicted boundary, yellow = agreement) plus a combined contact
sheet PNG, both under `outputs/overlays/`.

## Exporting predictions as GeoJSON (for the dashboard teammate)

```bash
python export_geojson.py --checkpoint checkpoints/best.pt --split test \
    --out outputs/predictions.geojson
```

Output schema — a standard `FeatureCollection`:

```json
{
  "type": "FeatureCollection",
  "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": [[[lon, lat], ...]]},
      "properties": {
        "field_id": "mock_field_00007",
        "polygon_index": 0,
        "source": "CTHBNet_prediction",
        "checkpoint": "best.pt",
        "threshold": 0.5,
        "georeferenced": true
      }
    }
  ]
}
```

**⚠️ Georeferencing gap to resolve with the preprocessing teammate:**
`CroplandDataset` as currently specified returns `(image, mask, field_id)`
with no per-chip affine transform / CRS. Without it, this script can only
export polygons in raw pixel `(col, row)` coordinates — it does this and
sets `"crs": null` plus a top-level `"_warning"` field so it's never
silently mistaken for lon/lat. To get real map coordinates, get either:
- a per-field affine transform (the standard `rasterio`/`affine`
  6-tuple `(a, b, c, d, e, f)`), or
- a lookup from `field_id` to that transform,

and pass it as `transform_lookup` to `export_predictions()` (see
`export_geojson.get_transform_for_field()` — that's the one function to
wire up).

## Loading a checkpoint for inference only

```python
import torch
from architecture import CTHBNet
import config

model = CTHBNet(
    in_channels=config.IN_CHANNELS, base_channels=config.BASE_CHANNELS,
    embed_dim=config.EMBED_DIM, depth=config.TRANSFORMER_DEPTH,
    num_heads=config.NUM_HEADS, dropout=config.DROPOUT,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

with torch.no_grad():
    logits = model(image_tensor.unsqueeze(0))   # (1, C, H, W) -> (1, 1, H, W)
    prob = torch.sigmoid(logits)
    pred_mask = (prob > 0.5).float()
```

(`weights_only=False` is needed because training checkpoints bundle
optimizer/scheduler/RNG state, not just weights — only load checkpoints
this repo's own `train.py` produced.)

## Notes / things verified

- Forward/backward pass, full training loop, checkpoint save, and
  checkpoint **resume** (including exact epoch continuation and RNG state)
  were run end-to-end against the mock dataset.
- Boundary extraction (used by both the loss and boundary-F1 metric) was
  verified against a known synthetic mask to confirm it correctly returns
  an empty boundary for a flat/constant region and a thin ring for a
  rectangular field mask.
- GeoJSON export was verified in both the pixel-coordinate fallback path
  and the georeferenced path (with a synthetic affine transform).
