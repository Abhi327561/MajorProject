"""
Exports predicted field boundary masks as GeoJSON polygons -- the format
the NDVI-dashboard teammate consumes to overlay boundaries on the map and
mask NDVI to individual fields.

Georeferencing note (read this before wiring to the real pipeline)
--------------------------------------------------------------------
`CroplandDataset` as currently specified returns `(image, mask, field_id)`
with no geotransform / CRS attached to each chip. Pixel-to-map coordinate
conversion needs *some* per-tile geospatial reference (an affine transform
+ CRS, e.g. what you'd get from a Sentinel-2 tile's `rasterio` profile).

This script supports both cases:
  1. If you pass a `transform_lookup` (dict: field_id -> affine transform)
     or the dataset yields it, polygons are exported in real-world
     coordinates (lon/lat, assuming the source transform is in a
     lon/lat-compatible CRS, or already reprojected).
  2. If no transform is available, polygons are exported in raw pixel
     coordinates (row/col of the chip) with a `"crs": null` field and a
     note in properties -- still useful for the dashboard team to test
     their overlay code before georeferencing is wired up, but flag this
     explicitly so it isn't silently treated as lon/lat.

Ask the preprocessing teammate for the per-chip affine transform (or a way
to look it up from `field_id`) and plug it into `get_transform_for_field`
below once available.

Usage
-----
    python export_geojson.py --checkpoint checkpoints/best.pt --split test \
        --out outputs/predictions.geojson
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage import measure
from shapely.geometry import Polygon, mapping
from shapely.validation import make_valid

import config
from architecture import CTHBNet
from dataset import get_datasets
from metrics import binarize


def get_transform_for_field(field_id, transform_lookup=None):
    """
    Returns an affine transform (a, b, c, d, e, f) mapping pixel (col, row)
    -> (x, y) in map coordinates, following the standard
    x = a*col + b*row + c ; y = d*col + e*row + f convention (same as
    rasterio/affine). Returns None if unavailable (falls back to pixel
    coordinates in the export).

    Wire this up to the preprocessing teammate's tile metadata once it's
    available, e.g.:
        return transform_lookup[field_id]
    """
    if transform_lookup is not None and field_id in transform_lookup:
        return transform_lookup[field_id]
    return None


def apply_transform(coords_rc, transform):
    """coords_rc: array of (row, col) pixel coords. Returns (x, y) map coords."""
    a, b, c, d, e, f = transform
    rows = coords_rc[:, 0]
    cols = coords_rc[:, 1]
    x = a * cols + b * rows + c
    y = d * cols + e * rows + f
    return np.stack([x, y], axis=1)


def mask_to_polygons(mask_2d, simplify_tolerance=1.0, min_area_px=9):
    """
    mask_2d: (H, W) binary numpy array.
    Returns a list of shapely Polygons (exterior ring only; holes are kept
    if present and large enough) extracted via marginal-square contouring,
    then simplified (Douglas-Peucker) to keep the GeoJSON compact.
    """
    contours = measure.find_contours(mask_2d.astype(float), level=0.5)
    polygons = []
    for contour in contours:
        # skimage gives (row, col); Polygon wants (x, y) -- caller decides
        # whether that's pixel (col, row) or map coords after transform.
        if len(contour) < 3:
            continue
        poly = Polygon(contour)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < min_area_px:
            continue
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        if poly.is_empty:
            continue
        polygons.append(poly)
    return polygons


@torch.no_grad()
def export_predictions(checkpoint_path, split="test", out_path=None,
                        transform_lookup=None, threshold=None,
                        simplify_tolerance=1.0):
    out_path = out_path or os.path.join(config.OUTPUT_DIR, f"predictions_{split}.geojson")
    threshold = threshold if threshold is not None else config.PRED_THRESHOLD

    device = torch.device(config.DEVICE)
    model = CTHBNet(
        in_channels=config.IN_CHANNELS, base_channels=config.BASE_CHANNELS,
        embed_dim=config.EMBED_DIM, depth=config.TRANSFORMER_DEPTH,
        num_heads=config.NUM_HEADS, dropout=config.DROPOUT,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_ds, val_ds, test_ds = get_datasets(use_mock=config.USE_MOCK_DATASET)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    features = []
    any_georeferenced = False

    for image, _mask, field_id in loader:
        field_id = field_id[0]
        image = image.to(device)
        logits = model(image)
        pred = binarize(logits, threshold=threshold, is_logits=True)[0, 0].cpu().numpy()

        polygons = mask_to_polygons(pred, simplify_tolerance=simplify_tolerance)
        transform = get_transform_for_field(field_id, transform_lookup)
        is_georeferenced = transform is not None
        any_georeferenced = any_georeferenced or is_georeferenced

        for poly_idx, poly in enumerate(polygons):
            coords_rc = np.array(poly.exterior.coords)  # (row, col)
            if is_georeferenced:
                coords_xy = apply_transform(coords_rc, transform)
            else:
                # fall back to (col, row) pixel coords as (x, y) so the
                # ring is still valid GeoJSON geometry, just not geo-real
                coords_xy = coords_rc[:, [1, 0]]

            geom = Polygon(coords_xy)
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "field_id": field_id,
                    "polygon_index": poly_idx,
                    "source": "CTHBNet_prediction",
                    "checkpoint": os.path.basename(checkpoint_path),
                    "threshold": threshold,
                    "georeferenced": is_georeferenced,
                },
            })

    geojson = {
        "type": "FeatureCollection",
        "crs": None if not any_georeferenced else {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    if not any_georeferenced:
        geojson["_warning"] = (
            "No affine transform was available for any field, so geometry "
            "coordinates are raw pixel (col, row) values, NOT lon/lat. "
            "Wire up get_transform_for_field() before handing this to the "
            "dashboard for map overlay."
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"Wrote {len(features)} polygon feature(s) to {out_path}")
    if not any_georeferenced:
        print("WARNING: exported in pixel coordinates -- see _warning field in the GeoJSON.")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Export predicted field boundaries as GeoJSON")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--simplify_tolerance", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_predictions(
        checkpoint_path=args.checkpoint, split=args.split, out_path=args.out,
        threshold=args.threshold, simplify_tolerance=args.simplify_tolerance,
    )
