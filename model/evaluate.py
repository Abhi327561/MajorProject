import torch
from torch.utils.data import DataLoader

import config
from architecture import CTHBNet
from losses import BoundaryAwareLoss
from metrics import compute_all_metrics, MetricAccumulator
from dataset import get_datasets


def main():

    device = torch.device(config.DEVICE)
    print("Device:", device)

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    _, _, test_ds = get_datasets(use_mock=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    print("Test samples:", len(test_ds))

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = CTHBNet(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        embed_dim=config.EMBED_DIM,
        depth=config.TRANSFORMER_DEPTH,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
    ).to(device)

    # ---------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------

    checkpoint_path = "checkpoints/best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"] + 1
    )

    print(
        "Best validation IoU:",
        checkpoint.get("best_metric", "N/A")
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    model.eval()

    criterion = BoundaryAwareLoss(
        w_bce=config.W_BCE,
        w_dice=config.W_DICE,
        w_boundary=config.W_BOUNDARY,
        boundary_dilate_px=config.BOUNDARY_DILATE_PX,
    )

    metric_acc = MetricAccumulator()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for images, masks, field_ids in test_loader:

            images = images.to(device)
            masks = masks.to(device).float()

            logits = model(images)

            loss, parts = criterion(logits, masks)

            total_loss += loss.item()
            num_batches += 1

            metrics = compute_all_metrics(
                logits,
                masks,
                threshold=config.PRED_THRESHOLD,
                boundary_tolerance=config.BOUNDARY_TOLERANCE_PX,
            )

            metric_acc.update(
                metrics,
                n=images.size(0)
            )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    avg_loss = total_loss / max(num_batches, 1)

    results = metric_acc.average()

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss          : {avg_loss:.4f}")
    print(f"IoU                : {results['iou']:.4f}")
    print(f"Precision          : {results['precision']:.4f}")
    print(f"Recall             : {results['recall']:.4f}")
    print(f"F1                 : {results['f1']:.4f}")
    print(f"Boundary Precision : {results['boundary_precision']:.4f}")
    print(f"Boundary Recall    : {results['boundary_recall']:.4f}")
    print(f"Boundary F1        : {results['boundary_f1']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()