"""Generates realistic Sentinel-2 false-color NDVI crop imagery thumbnails for cropland monitoring dashboard."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import json

def generate_all_thumbnails():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imagery_dir = os.path.join(base_dir, "data", "imagery")
    os.makedirs(imagery_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, "data", "mock_ndvi_timeseries.csv")
    geojson_path = os.path.join(base_dir, "data", "mock_fields.geojson")
    
    if not os.path.exists(csv_path) or not os.path.exists(geojson_path):
        print("Data files not found.")
        return
        
    df = pd.read_csv(csv_path)
    with open(geojson_path, "r") as f:
        geojson = json.load(f)
        
    fields = {feat["properties"]["field_id"]: feat["properties"]["field_name"] for feat in geojson["features"]}
    
    for _, row in df.iterrows():
        field_id = str(row["field_id"])
        date_str = str(row["acquisition_date"])
        ndvi_val = float(row["ndvi_mean"])
        cloud_val = float(row["cloud_cover"])
        
        img_filename = f"{field_id}_{date_str}.png"
        img_filepath = os.path.join(imagery_dir, img_filename)
        
        # Generate synthetic Sentinel false color image (256x256)
        np.random.seed(int(field_id[-2:]) * 100 + int(date_str.split("-")[1]))
        
        # Create base canopy texture
        size = 256
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        
        # Base background noise
        bg = np.random.normal(0.3, 0.05, (size, size))
        
        # Field mask in center
        mask = (xx > -0.65) & (xx < 0.65) & (yy > -0.65) & (yy < 0.65)
        
        # Crop canopy with NDVI variance
        field_pattern = np.sin(xx * 20) * 0.03 + np.cos(yy * 15) * 0.03
        canopy_ndvi = np.clip(ndvi_val + field_pattern + np.random.normal(0, 0.03, (size, size)), 0.05, 0.95)
        
        full_ndvi = np.where(mask, canopy_ndvi, bg)
        
        # Add clouds if cloudy
        if cloud_val > 20:
            cloud_blob = np.exp(-((xx - 0.2)**2 + (yy - 0.3)**2) / 0.25) * (cloud_val / 100.0)
            full_ndvi = np.clip(full_ndvi * (1 - cloud_blob) + cloud_blob * 0.1, 0, 1)
            
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        im = ax.imshow(full_ndvi, cmap="RdYlGn", vmin=0.0, vmax=0.9)
        
        # Draw field boundary rectangle
        rect = patches.Rectangle(
            (size * 0.175, size * 0.175), size * 0.65, size * 0.65,
            linewidth=2.5, edgecolor="cyan", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        
        # Title and details overlay
        ax.text(
            12, 25, f"{fields.get(field_id, field_id)} | {date_str}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
        )
        ax.text(
            12, size - 15, f"NDVI: {ndvi_val:.2f} | Cloud: {cloud_val:.1f}%",
            color="white", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
        )
        
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(img_filepath, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        
    print(f"Generated {len(df)} thumbnail images in {imagery_dir}")

if __name__ == "__main__":
    generate_all_thumbnails()
