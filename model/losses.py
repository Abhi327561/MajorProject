"""
Combined loss for field boundary segmentation:

    total = w_bce * BCE  +  w_dice * Dice  +  w_boundary * BoundaryLoss

- BCE + Dice: standard region-overlap segmentation losses.
- BoundaryLoss: penalizes errors specifically on/near field edges, extracted
  from the ground-truth mask with a Sobel filter, so the model is pushed to
  get boundary pixels right even when they're a tiny fraction of the image
  (field interiors dominate pixel count otherwise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def extract_boundary(mask, dilate_px=1):
    """
    mask: (B, 1, H, W) binary {0,1} float tensor.
    Returns a soft boundary map (B, 1, H, W) with edges highlighted via
    Sobel gradient magnitude, optionally dilated a couple pixels so the
    boundary band has non-trivial width for the loss to act on.
    """
    kx, ky = _sobel_kernels(mask.device, mask.dtype)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    # No epsilon here: this map only ever feeds a hard threshold (never
    # backpropagated through), and adding eps under the sqrt makes flat
    # regions register as "boundary" once compared with `> 0`.
    grad = torch.sqrt(gx ** 2 + gy ** 2)
    grad = (grad > 1e-4).float()
    if dilate_px > 0:
        grad = F.max_pool2d(grad, kernel_size=2 * dilate_px + 1, stride=1, padding=dilate_px)
    return grad


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.flatten(1)
        target = target.flatten(1)
        intersection = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Weighted BCE between predicted probability and a Sobel-derived boundary
    map of the ground truth, focusing gradient signal on edge pixels.
    """

    def __init__(self, dilate_px=1):
        super().__init__()
        self.dilate_px = dilate_px

    def forward(self, logits, target):
        with torch.no_grad():
            boundary_target = extract_boundary(target, self.dilate_px)
        # Compare predicted probability directly (differentiable) against the
        # boundary band derived from ground truth, rather than trying to
        # differentiate through a Sobel filter on the prediction itself.
        prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(prob, target, reduction="none")
        # Emphasize loss on the boundary band; de-emphasize interior/background.
        weight = 1.0 + 4.0 * boundary_target
        return (loss * weight).mean()


class BoundaryAwareLoss(nn.Module):
    """Combined segmentation + boundary loss for field boundary prediction."""

    def __init__(self, w_bce=1.0, w_dice=1.0, w_boundary=1.0, boundary_dilate_px=1):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(dilate_px=boundary_dilate_px)

    def forward(self, logits, target):
        """
        logits: (B, 1, H, W) raw model output
        target: (B, 1, H, W) binary ground-truth mask, float {0., 1.}
        Returns: (total_loss, dict of component losses for logging)
        """
        target = target.float()
        bce_l = self.bce(logits, target)
        dice_l = self.dice(logits, target)
        boundary_l = self.boundary(logits, target)
        total = self.w_bce * bce_l + self.w_dice * dice_l + self.w_boundary * boundary_l
        return total, {
            "bce": bce_l.item(),
            "dice": dice_l.item(),
            "boundary": boundary_l.item(),
            "total": total.item(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()
    criterion = BoundaryAwareLoss()
    total, parts = criterion(logits, target)
    print("total:", total.item())
    print(parts)
