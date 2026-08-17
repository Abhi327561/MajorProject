"""
Evaluation metrics for field boundary segmentation.

- IoU (region-level): intersection-over-union of the full predicted field
  mask vs. ground truth.
- Precision / Recall / F1 (region-level): standard pixel classification
  metrics over the full mask.
- Boundary F1: precision/recall/F1 computed only on the thin boundary band
  around field edges (via the same Sobel extraction used in the boundary
  loss), with a tolerance radius -- a predicted boundary pixel counts as
  correct if it falls within `tolerance` pixels of a ground-truth boundary
  pixel. This is the metric that actually reflects boundary sharpness,
  since region IoU can look fine even with blurry/offset edges.
"""

import torch
import torch.nn.functional as F

from losses import extract_boundary


@torch.no_grad()
def binarize(logits_or_prob, threshold=0.5, is_logits=True):
    prob = torch.sigmoid(logits_or_prob) if is_logits else logits_or_prob
    return (prob > threshold).float()


@torch.no_grad()
def iou_score(pred_mask, target_mask, eps=1e-6):
    """pred_mask, target_mask: (B, 1, H, W) binary {0,1}."""
    pred = pred_mask.flatten(1)
    target = target_mask.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


@torch.no_grad()
def precision_recall_f1(pred_mask, target_mask, eps=1e-6):
    pred = pred_mask.flatten(1)
    target = target_mask.flatten(1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision.mean().item(), recall.mean().item(), f1.mean().item()


@torch.no_grad()
def _dilate(mask, radius):
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


@torch.no_grad()
def boundary_f1_score(pred_mask, target_mask, tolerance=2):
    """
    Boundary F1 with a pixel tolerance radius (a common relaxation used for
    thin-structure segmentation, similar in spirit to BF-score).

    pred_mask, target_mask: (B, 1, H, W) binary {0,1} full-region masks.
    """
    pred_b = extract_boundary(pred_mask, dilate_px=0)
    target_b = extract_boundary(target_mask, dilate_px=0)

    target_b_tol = _dilate(target_b, tolerance)
    pred_b_tol = _dilate(pred_b, tolerance)

    pred_flat = pred_b.flatten(1)
    target_flat = target_b.flatten(1)
    pred_tol_flat = pred_b_tol.flatten(1)
    target_tol_flat = target_b_tol.flatten(1)

    eps = 1e-6
    # precision: fraction of predicted boundary pixels that lie within
    # `tolerance` of a true boundary pixel
    tp_p = (pred_flat * target_tol_flat).sum(dim=1)
    precision = (tp_p + eps) / (pred_flat.sum(dim=1) + eps)

    # recall: fraction of true boundary pixels that lie within `tolerance`
    # of a predicted boundary pixel
    tp_r = (target_flat * pred_tol_flat).sum(dim=1)
    recall = (tp_r + eps) / (target_flat.sum(dim=1) + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision.mean().item(), recall.mean().item(), f1.mean().item()


@torch.no_grad()
def compute_all_metrics(logits, target_mask, threshold=0.5, boundary_tolerance=2):
    """
    logits: (B, 1, H, W) raw model output
    target_mask: (B, 1, H, W) binary ground truth
    Returns a dict of scalar metrics (batch-averaged).
    """
    pred_mask = binarize(logits, threshold=threshold, is_logits=True)
    target_mask = target_mask.float()

    iou = iou_score(pred_mask, target_mask)
    precision, recall, f1 = precision_recall_f1(pred_mask, target_mask)
    b_precision, b_recall, b_f1 = boundary_f1_score(pred_mask, target_mask, tolerance=boundary_tolerance)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "boundary_precision": b_precision,
        "boundary_recall": b_recall,
        "boundary_f1": b_f1,
    }


class MetricAccumulator:
    """Running average of metric dicts across batches, for epoch-level logging."""

    def __init__(self):
        self.sums = {}
        self.count = 0

    def update(self, metrics_dict, n=1):
        for k, v in metrics_dict.items():
            self.sums[k] = self.sums.get(k, 0.0) + v * n
        self.count += n

    def average(self):
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.sums.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 64, 64) * 3
    target = (torch.rand(4, 1, 64, 64) > 0.6).float()
    m = compute_all_metrics(logits, target)
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
