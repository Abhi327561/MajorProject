"""
Generates predicted-vs-ground-truth boundary overlay images for the review
deck: for each sample, ground-truth boundary in one color, predicted
boundary in another, drawn over the RGB composite of the input.

Usage
-----
    python visualize.py --checkpoint checkpoints/best.pt --num_samples 6
    python visualize.py --checkpoint checkpoints/best.pt --split test --out outputs/overlays

Output: one PNG per sample under `--out` (default: outputs/overlays/), plus
a single contact-sheet PNG combining all samples for quick inclusion in
slides.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from architecture import CTHBNet
from dataset import get_datasets
from losses import extract_boundary
from metrics import binarize


def to_rgb_composite(image_tensor):
    """
    image_tensor: (C, H, W) with C >= 3, assumed first 3 channels are R,G,B
    (or a proxy). Normalizes per-channel to [0, 1] for display only.
    """
    img = image_tensor[:3].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img - img.min()
    denom = img.max() if img.max() > 1e-6 else 1.0
    img = img / denom
    return np.clip(img, 0, 1)


def boundary_pixels(mask_tensor):
    """mask_tensor: (1, H, W) binary. Returns (H, W) boundary map as numpy bool."""
    b = extract_boundary(mask_tensor.unsqueeze(0), dilate_px=1)[0, 0]
    return (b.detach().cpu().numpy() > 0)


def make_overlay(image_tensor, gt_mask, pred_mask):
    """Returns an (H, W, 3) RGB image with GT boundary in cyan, predicted boundary in magenta."""
    base = to_rgb_composite(image_tensor)
    overlay = base.copy()

    gt_b = boundary_pixels(gt_mask)
    pred_b = boundary_pixels(pred_mask)

    overlay[gt_b] = [0.0, 1.0, 1.0]      # cyan = ground truth
    overlay[pred_b] = [1.0, 0.0, 1.0]    # magenta = prediction
    both = gt_b & pred_b
    overlay[both] = [1.0, 1.0, 0.0]      # yellow = agreement

    return overlay


@torch.no_grad()
def generate_overlays(checkpoint_path, split="val", num_samples=6, out_dir=None, threshold=None):
    out_dir = out_dir or os.path.join(config.OUTPUT_DIR, "overlays")
    os.makedirs(out_dir, exist_ok=True)
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

    saved_paths = []
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]

    for i, (image, mask, field_id) in enumerate(loader):
        if i >= num_samples:
            break
        image = image.to(device)
        mask = mask.to(device)
        logits = model(image)
        pred_mask = binarize(logits, threshold=threshold, is_logits=True)

        overlay = make_overlay(image[0], mask[0], pred_mask[0])

        # per-sample PNG
        fname = f"overlay_{split}_{field_id[0]}.png"
        fpath = os.path.join(out_dir, fname)
        plt.imsave(fpath, overlay)
        saved_paths.append(fpath)

        axes[i].imshow(overlay)
        axes[i].set_title(field_id[0], fontsize=9)
        axes[i].axis("off")

    legend_text = "cyan = ground truth   magenta = prediction   yellow = agreement"
    fig.suptitle(f"Boundary overlays ({split}) -- {legend_text}", fontsize=10)
    fig.tight_layout()
    contact_sheet_path = os.path.join(out_dir, f"contact_sheet_{split}.png")
    fig.savefig(contact_sheet_path, dpi=150)
    plt.close(fig)

    print(f"Saved {len(saved_paths)} overlay(s) to {out_dir}")
    print(f"Contact sheet: {contact_sheet_path}")
    return saved_paths, contact_sheet_path


def parse_args():
    p = argparse.ArgumentParser(description="Generate predicted-vs-GT boundary overlays")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_overlays(
        checkpoint_path=args.checkpoint, split=args.split,
        num_samples=args.num_samples, out_dir=args.out, threshold=args.threshold,
    )
