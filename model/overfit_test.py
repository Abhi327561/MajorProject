import torch
from torch.utils.data import DataLoader, Subset

from dataset import AI4BoundariesDataset
from architecture import CTHBNet
from losses import BoundaryAwareLoss
from metrics import compute_all_metrics


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 8
EPOCHS = 20
BATCH_SIZE = 2
LR = 1e-4


print("Device:", DEVICE)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

full_ds = AI4BoundariesDataset(split="train")

# Only first 8 REAL samples
ds = Subset(full_ds, range(NUM_SAMPLES))

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


print("Overfit samples:", len(ds))


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = CTHBNet(
    in_channels=4
).to(DEVICE)


criterion = BoundaryAwareLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for images, masks, field_ids in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss, parts = criterion(
            logits,
            masks
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()


    # -----------------------------------------------------
    # Evaluate on the SAME 8 samples
    # -----------------------------------------------------

    model.eval()

    metric_sum = {}

    with torch.no_grad():

        for images, masks, field_ids in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            metrics = compute_all_metrics(
                logits,
                masks,
                threshold=0.5,
                boundary_tolerance=2,
            )

            for key, value in metrics.items():
                metric_sum[key] = metric_sum.get(key, 0.0) + value

    num_batches = len(loader)

    metrics = {
        key: value / num_batches
        for key, value in metric_sum.items()
    }

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"loss={avg_loss:.4f} "
        f"IoU={metrics['iou']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"BoundaryF1={metrics['boundary_f1']:.4f}"
    )


print("\nOverfit test complete.")