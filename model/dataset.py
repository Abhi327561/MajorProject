"""
Real AI4Boundaries dataset loader for CTHBNet.

Sentinel-2:
    C:/Users/mpabh/OneDrive/Desktop/srcdata/sentinel2/AT/*.nc

Masks:
    C:/Users/mpabh/OneDrive/Desktop/AI4Boundaries/masks/{train,val,test}/*.tif

Each Sentinel .nc contains:
    B2  : (time, 256, 256)
    B3  : (time, 256, 256)
    B4  : (time, 256, 256)
    B8  : (time, 256, 256)
    NDVI: (time, 256, 256)

Each TIFF contains 4 bands:
    Band 1 -> field/parcel mask
    Band 2 -> boundary mask
    Band 3 -> auxiliary information
    Band 4 -> auxiliary information

For CTHBNet:
    Input  -> B2, B3, B4, B8
    Target -> TIFF Band 1

Each month is treated as a separate observation.
Therefore:
    43 Sentinel files × 6 months = up to 258 samples.
"""

import os
import glob

import numpy as np
import torch
import xarray as xr
import rasterio
from torch.utils.data import Dataset


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

SENTINEL_DIR = r"C:\Users\mpabh\OneDrive\Desktop\srcdata\sentinel2\AT"

MASK_ROOT = r"C:\Users\mpabh\OneDrive\Desktop\AI4Boundaries\masks"


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class AI4BoundariesDataset(Dataset):

    def __init__(
        self,
        split="train",
        sentinel_dir=SENTINEL_DIR,
        mask_root=MASK_ROOT,
    ):
        self.split = split
        self.sentinel_dir = sentinel_dir
        self.mask_dir = os.path.join(mask_root, split)

        if not os.path.isdir(self.sentinel_dir):
            raise FileNotFoundError(
                f"Sentinel directory not found:\n{self.sentinel_dir}"
            )

        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"Mask directory not found:\n{self.mask_dir}"
            )

        # -------------------------------------------------------------
        # Find Sentinel files
        # -------------------------------------------------------------

        sentinel_files = sorted(
            glob.glob(os.path.join(self.sentinel_dir, "*.nc"))
        )

        if not sentinel_files:
            raise RuntimeError(
                f"No .nc files found in:\n{self.sentinel_dir}"
            )

        # -------------------------------------------------------------
        # Build valid pairs
        # -------------------------------------------------------------

        self.samples = []

        for nc_path in sentinel_files:

            base_name = os.path.splitext(
                os.path.basename(nc_path)
            )[0]

            mask_path = os.path.join(
                self.mask_dir,
                base_name + ".tif"
            )

            if not os.path.exists(mask_path):
                continue

            # ---------------------------------------------------------
            # Determine number of time steps
            # ---------------------------------------------------------

            with xr.open_dataset(nc_path) as ds:

                if "time" not in ds.dims:
                    raise ValueError(
                        f"No time dimension found in {nc_path}"
                    )

                num_times = ds.sizes["time"]

            # ---------------------------------------------------------
            # Create one sample per month
            # ---------------------------------------------------------

            for time_idx in range(num_times):

                self.samples.append(
                    {
                        "nc_path": nc_path,
                        "mask_path": mask_path,
                        "time_idx": time_idx,
                        "field_id": base_name,
                    }
                )

        if not self.samples:
            raise RuntimeError(
                f"No matching Sentinel/mask pairs found for split '{split}'."
            )

        print(
            f"[AI4BoundariesDataset] split={split} "
            f"pairs={len(set(s['nc_path'] for s in self.samples))} "
            f"samples={len(self.samples)}"
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # Sentinel loading
    # ------------------------------------------------------------------

    def _load_sentinel(self, nc_path, time_idx):

        with xr.open_dataset(nc_path) as ds:

            bands = []

            for band_name in ["B2", "B3", "B4", "B8"]:

                if band_name not in ds:
                    raise KeyError(
                        f"{band_name} not found in {nc_path}"
                    )

                arr = ds[band_name].isel(
                    time=time_idx
                ).values.astype(np.float32)

                # Sentinel nodata
                arr[arr == -9999] = 0.0

                # Convert scaled reflectance to approximately [0,1]
                arr = arr / 10000.0

                # Keep model input numerically stable
                arr = np.clip(arr, 0.0, 1.0)

                bands.append(arr)

        image = np.stack(bands, axis=0)

        return image

    # ------------------------------------------------------------------
    # Mask loading
    # ------------------------------------------------------------------

    def _load_mask(self, mask_path):

        with rasterio.open(mask_path) as src:

            # Band 1 = field / parcel mask
            field_mask = src.read(1).astype(np.float32)

            # Band 2 = real boundary annotation
            boundary_mask = src.read(2).astype(np.float32)

        field_mask = (field_mask > 0.5).astype(np.float32)
        boundary_mask = (boundary_mask > 0.5).astype(np.float32)

        return field_mask, boundary_mask

    # ------------------------------------------------------------------
    # Get item
    # ------------------------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = self._load_sentinel(
            sample["nc_path"],
            sample["time_idx"]
        )

        field_mask, boundary_mask = self._load_mask(
            sample["mask_path"]
        )

        # -------------------------------------------------------------
        # Shape validation
        # -------------------------------------------------------------

        if image.shape != (4, 256, 256):
            raise ValueError(
                f"Unexpected image shape: {image.shape}"
            )

        if field_mask.shape != (256, 256):
            raise ValueError(
                f"Unexpected mask shape: {field_mask.shape}"
            )

        # -------------------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------------------

        image_tensor = torch.from_numpy(image).float()

        mask_tensor = torch.from_numpy(
            field_mask
        ).unsqueeze(0).float()

        field_id = (
            f"{sample['field_id']}"
            f"_t{sample['time_idx']}"
        )

        return image_tensor, mask_tensor, field_id


# -------------------------------------------------------------------------
# Dataset factory
# -------------------------------------------------------------------------

def get_datasets(use_mock=False, **kwargs):

    if use_mock:
        raise RuntimeError(
            "Mock dataset is disabled for the real AI4Boundaries run."
        )

    train_ds = AI4BoundariesDataset(
        split="train",
        **kwargs
    )

    val_ds = AI4BoundariesDataset(
        split="val",
        **kwargs
    )

    test_ds = AI4BoundariesDataset(
        split="test",
        **kwargs
    )

    return train_ds, val_ds, test_ds


# -------------------------------------------------------------------------
# Standalone sanity test
# -------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nTesting AI4Boundaries dataset...\n")

    ds = AI4BoundariesDataset(split="train")

    print("\nDataset length:", len(ds))

    image, mask, field_id = ds[0]

    print("\nFirst sample:")
    print("field_id:", field_id)
    print("image shape:", image.shape)
    print("image dtype:", image.dtype)
    print(
        "image range:",
        float(image.min()),
        "to",
        float(image.max())
    )

    print("mask shape:", mask.shape)
    print("mask dtype:", mask.dtype)
    print("mask unique:", torch.unique(mask))
    print(
        "mask foreground:",
        int(mask.sum()),
        "/",
        mask.numel()
    )

    print("\nDataset test successful.")