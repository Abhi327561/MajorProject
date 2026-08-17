"""
Training script for CTHBNet field boundary segmentation.

Usage
-----
    python train.py                       # fresh run, mock dataset, defaults from config.py
    python train.py --epochs 30 --lr 1e-4
    python train.py --resume checkpoints/last.pt      # resume a Colab session that got cut off
    python train.py --no-mock                          # once CroplandDataset is wired in

Checkpointing / resume
-----------------------
Every epoch we save:
    checkpoints/last.pt         -- always overwritten, latest state (for resume)
    checkpoints/best.pt         -- overwritten only when val boundary_f1 improves
    checkpoints/epoch_XXX.pt    -- optional periodic snapshot (every `--ckpt_every` epochs)

A checkpoint contains model, optimizer, scheduler, epoch number, best metric
so far, and the RNG state, so `--resume` continues training exactly where it
left off (important on Colab where sessions disconnect mid-run).
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from architecture import CTHBNet
from losses import BoundaryAwareLoss
from metrics import compute_all_metrics, MetricAccumulator
from dataset import get_datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, args):
    if config.LR_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif config.LR_SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
    raise ValueError(f"Unknown LR_SCHEDULER: {config.LR_SCHEDULER}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_metric": best_metric,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "args": vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    # weights_only=False: our checkpoints bundle optimizer/scheduler/RNG state
    # (not just tensors), so this needs the full unpickler. Only load
    # checkpoints you trust (i.e. ones this training run produced).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    rng = ckpt.get("rng_state")
    if rng is not None:
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    start_epoch = ckpt["epoch"] + 1
    best_metric = ckpt.get("best_metric", -1.0)
    print(f"Resumed from '{path}' -> starting at epoch {start_epoch}, best_metric so far = {best_metric:.4f}")
    return start_epoch, best_metric


def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=None):
    model.train(mode=train)
    loss_acc = 0.0
    metric_acc = MetricAccumulator()
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks, _field_ids in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            logits = model(images)
            loss, _parts = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            loss_acc += loss.item()
            n_batches += 1
            batch_metrics = compute_all_metrics(
                logits.detach(), masks, threshold=config.PRED_THRESHOLD,
                boundary_tolerance=config.BOUNDARY_TOLERANCE_PX,
            )
            metric_acc.update(batch_metrics, n=images.size(0))

    avg_loss = loss_acc / max(n_batches, 1)
    avg_metrics = metric_acc.average()
    return avg_loss, avg_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train CTHBNet for field boundary segmentation")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    p.add_argument("--ckpt_every", type=int, default=1, help="save a numbered snapshot every N epochs (0 to disable)")
    p.add_argument("--no-mock", action="store_true", help="use the real CroplandDataset instead of the mock")
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--device", type=str, default=config.DEVICE)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_ds, val_ds, _test_ds = get_datasets(use_mock=not args.no_mock)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = CTHBNet(
        in_channels=config.IN_CHANNELS, base_channels=config.BASE_CHANNELS,
        embed_dim=config.EMBED_DIM, depth=config.TRANSFORMER_DEPTH,
        num_heads=config.NUM_HEADS, dropout=config.DROPOUT,
    ).to(device)

    criterion = BoundaryAwareLoss(
        w_bce=config.W_BCE, w_dice=config.W_DICE, w_boundary=config.W_BOUNDARY,
        boundary_dilate_px=config.BOUNDARY_DILATE_PX,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)

    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        start_epoch, best_metric = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    last_path = os.path.join(config.CHECKPOINT_DIR, "last.pt")
    best_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            grad_clip=config.GRAD_CLIP_NORM,
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
        )
        scheduler.step()
        dt = time.time() - t0

        print(
    f"[epoch {epoch+1:03d}/{args.epochs}] "
    f"train_loss={train_loss:.4f} "
    f"val_loss={val_loss:.4f} "
    f"val_iou={val_metrics.get('iou', 0):.4f} "
    f"val_precision={val_metrics.get('precision', 0):.4f} "
    f"val_recall={val_metrics.get('recall', 0):.4f} "
    f"val_f1={val_metrics.get('f1', 0):.4f} "
    f"val_boundary_f1={val_metrics.get('boundary_f1', 0):.4f} "
    f"lr={scheduler.get_last_lr()[0]:.2e} "
    f"({dt:.1f}s)"
)

        # Always update "last" so a disconnected Colab session can resume.
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_metric, args)

        current_metric = val_metrics.get("iou", 0.0)

        if current_metric > best_metric:
            best_metric = current_metric

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                args,
            )

            print(
                f"  -> new best (val_iou={best_metric:.4f}), "
                f"saved to {best_path}"
            )

        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
            snap_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pt")
            save_checkpoint(snap_path, model, optimizer, scheduler, epoch, best_metric, args)

    print("Training complete. Best val_iou:", best_metric)


if __name__ == "__main__":
    main()
