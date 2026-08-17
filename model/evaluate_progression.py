"""
Re-evaluates every saved epoch checkpoint (checkpoints/epoch_XXX.pt) against
the val (and optionally test) split, and prints/saves a table of metrics
per epoch.
"""

import argparse
import csv
import glob
import os
import re

import torch
from torch.utils.data import DataLoader

import config
from architecture import CTHBNet
from dataset import get_datasets
from metrics import compute_all_metrics, MetricAccumulator


def find_epoch_checkpoints(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))

    def epoch_num(p):
        m = re.search(r"epoch_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return sorted(paths, key=epoch_num)


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, loader, device):
    model = CTHBNet(
        in_channels=config.IN_CHANNELS, base_channels=config.BASE_CHANNELS,
        embed_dim=config.EMBED_DIM, depth=config.TRANSFORMER_DEPTH,
        num_heads=config.NUM_HEADS, dropout=config.DROPOUT,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    acc = MetricAccumulator()
    for images, masks, _field_ids in loader:
        images = images.to(device)
        masks = masks.to(device).float()
        logits = model(images)
        m = compute_all_metrics(
            logits, masks, threshold=config.PRED_THRESHOLD,
            boundary_tolerance=config.BOUNDARY_TOLERANCE_PX,
        )
        acc.update(m, n=images.size(0))

    epoch = ckpt.get("epoch", -1) + 1
    return epoch, acc.average()


def main():
    p = argparse.ArgumentParser(description="Evaluate the metric progression across saved checkpoints")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--ckpt_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--checkpoints", type=str, nargs="*", default=None)
    p.add_argument("--out", type=str, default=os.path.join(config.OUTPUT_DIR, "progression.csv"))
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--no-mock", action="store_true")
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    train_ds, val_ds, test_ds = get_datasets(use_mock=not args.no_mock)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt_paths = args.checkpoints or find_epoch_checkpoints(args.ckpt_dir)
    if not ckpt_paths:
        print(f"No epoch_*.pt checkpoints found in {args.ckpt_dir}.")
        return

    rows = []
    print(f"Evaluating {len(ckpt_paths)} checkpoint(s) on '{args.split}' split...\n")
    header = f"{'epoch':>6} | {'iou':>7} | {'prec':>7} | {'recall':>7} | {'f1':>7} | {'b_prec':>7} | {'b_recall':>8} | {'b_f1':>7}"
    print(header)
    print("-" * len(header))

    for ckpt_path in ckpt_paths:
        epoch, m = evaluate_checkpoint(ckpt_path, loader, device)
        row = {
            "epoch": epoch, "checkpoint": os.path.basename(ckpt_path),
            "iou": m["iou"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "boundary_precision": m["boundary_precision"], "boundary_recall": m["boundary_recall"],
            "boundary_f1": m["boundary_f1"],
        }
        rows.append(row)
        print(f"{epoch:>6} | {m['iou']:>7.4f} | {m['precision']:>7.4f} | {m['recall']:>7.4f} | "
              f"{m['f1']:>7.4f} | {m['boundary_precision']:>7.4f} | {m['boundary_recall']:>8.4f} | {m['boundary_f1']:>7.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved progression table to {args.out}")

    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print("\n--- Trend summary (first checkpoint -> last checkpoint) ---")
        for key in ["iou", "boundary_f1", "precision", "recall"]:
            delta = last[key] - first[key]
            arrow = "up" if delta > 1e-4 else ("down" if delta < -1e-4 else "flat")
            print(f"{key:>18}: {first[key]:.4f} -> {last[key]:.4f}  ({arrow}, delta={delta:+.4f})")


if __name__ == "__main__":
    main()
