"""
CTHBNet-style CNN-Transformer Hybrid for field boundary segmentation.

Design
------
- CNNBranch: a lightweight U-Net-style encoder (4 downsampling stages) that
  captures local edge / texture detail. Skip connections are kept for the
  decoder.
- TransformerBranch: patchifies the deepest CNN feature map, runs it through
  a standard ViT-style encoder (multi-head self-attention + MLP blocks) to
  capture global field-scale context (shape, neighboring-field relations),
  then reshapes back to a spatial map.
- FusionModule: concatenates the CNN's deepest features with the
  Transformer's global-context features and mixes them with a 1x1 conv.
- Decoder: progressively upsamples the fused features back to full
  resolution using the CNN encoder's skip connections (U-Net-style),
  producing a single-channel per-pixel boundary logit map.

Input:  (B, C_in, H, W)  -- default C_in=4 (e.g. Sentinel-2 R,G,B,NIR bands)
Output: (B, 1, H, W)     -- raw logits (apply sigmoid for probability)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNBranch(nn.Module):
    """4-stage encoder capturing local detail. Returns skip features + deepest feature map."""

    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        c = base_channels
        self.enc1 = conv_block(in_channels, c)          # H, W
        self.enc2 = conv_block(c, c * 2)                # H/2, W/2
        self.enc3 = conv_block(c * 2, c * 4)             # H/4, W/4
        self.enc4 = conv_block(c * 4, c * 8)             # H/8, W/8
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.enc1(x)                 # (B, c,   H,    W)
        s2 = self.enc2(self.pool(s1))     # (B, 2c,  H/2,  W/2)
        s3 = self.enc3(self.pool(s2))     # (B, 4c,  H/4,  W/4)
        s4 = self.enc4(self.pool(s3))     # (B, 8c,  H/8,  W/8)  <- deepest / local features
        return s1, s2, s3, s4


class TransformerBranch(nn.Module):
    """
    Patchifies the CNN's deepest feature map and runs a ViT-style encoder
    to capture global context, then projects back to a spatial map of the
    same shape as the input feature map.
    """

    def __init__(self, in_channels, embed_dim=256, depth=4, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1, max_tokens=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        self.max_tokens = max_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def _interpolated_pos_embed(self, num_tokens, h, w, device):
        # Bilinearly resize the learned positional grid to the current H x W
        # so the branch generalizes to inputs of varying spatial size.
        base_hw = int(math.sqrt(self.max_tokens))
        pe = self.pos_embed.reshape(1, base_hw, base_hw, self.embed_dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, num_tokens, self.embed_dim)
        return pe.to(device)

    def forward(self, feat):
        B, C, H, W = feat.shape
        x = self.proj_in(feat)                       # (B, E, H, W)
        tokens = x.flatten(2).transpose(1, 2)         # (B, H*W, E)
        tokens = tokens + self._interpolated_pos_embed(H * W, H, W, feat.device)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        out = tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        out = self.proj_out(out)                      # back to (B, C, H, W)
        return out


class FusionModule(nn.Module):
    """Combines CNN local features with Transformer global-context features."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # learnable gate so the network can weight local vs global signal
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, cnn_feat, trans_feat):
        combined = torch.cat([cnn_feat, trans_feat], dim=1)
        gate = self.gate(combined)
        fused = self.mix(combined)
        return fused * gate + cnn_feat * (1 - gate)


def up_block(in_ch, skip_ch, out_ch):
    return nn.ModuleDict({
        "up": nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        "conv": conv_block(out_ch + skip_ch, out_ch),
    })


class Decoder(nn.Module):
    """U-Net-style decoder that upsamples fused features back to full resolution."""

    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        self.up3 = up_block(c * 8, c * 4, c * 4)
        self.up2 = up_block(c * 4, c * 2, c * 2)
        self.up1 = up_block(c * 2, c * 1, c * 1)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def _up_and_concat(self, block, x, skip):
        x = block["up"](x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return block["conv"](x)

    def forward(self, fused, s1, s2, s3):
        x = self._up_and_concat(self.up3, fused, s3)
        x = self._up_and_concat(self.up2, x, s2)
        x = self._up_and_concat(self.up1, x, s1)
        return self.head(x)   # (B, 1, H, W) raw logits


class CTHBNet(nn.Module):
    """
    CNN-Transformer Hybrid for field boundary segmentation.

    Args:
        in_channels: number of input spectral bands (default 4: R,G,B,NIR).
        base_channels: base width of the CNN encoder.
        embed_dim: token embedding dim for the transformer branch.
        depth: number of transformer encoder layers.
        num_heads: attention heads.
    """

    def __init__(self, in_channels=4, base_channels=64, embed_dim=256,
                 depth=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.cnn = CNNBranch(in_channels, base_channels)
        deepest_ch = base_channels * 8
        self.transformer = TransformerBranch(
            in_channels=deepest_ch, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, dropout=dropout,
        )
        self.fusion = FusionModule(deepest_ch)
        self.decoder = Decoder(base_channels)

    def forward(self, x):
        s1, s2, s3, s4 = self.cnn(x)
        global_ctx = self.transformer(s4)
        fused = self.fusion(s4, global_ctx)
        logits = self.decoder(fused, s1, s2, s3)
        return logits  # (B, 1, H, W) -- apply sigmoid outside for probabilities


if __name__ == "__main__":
    model = CTHBNet(in_channels=4)
    x = torch.randn(2, 4, 256, 256)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", out.shape)
    print(f"Total parameters: {n_params:,}")
