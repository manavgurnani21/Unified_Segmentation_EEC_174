"""CLRKD-style curve lane head used by the in-house Stage 2 fusion model.

This is intentionally lighter than the upstream CLRHead. It mirrors the
prior representation but does not replicate the iterative ROI refinement
loop. The vendored CLRKDNet path is preserved separately for full-paper
reproductions; this head is what the fusion experiments (3, 4, 5) train.

Output API (per image):
    cls_logits  : (B, max_lanes)               existence logit
    coord_pred  : (B, max_lanes, num_points, 2) normalized (x, y) in [0, 1]
    mask_logit  : (B, 1, H, W)                  optional auxiliary mask
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_act(c_in: int, c_out: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CurveLaneHead(nn.Module):
    """Predict (existence, points) for a fixed number of lane slots.

    Inputs : list of feature maps with channels and strides given by
             `in_channels` and decreasing strides (highest-res last).
    Outputs: dict with keys cls_logits, coord_pred, mask_logit.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 256, 256),
        embed_dim: int = 128,
        max_lanes: int = 10,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = True,
        num_lane_classes: int = 7,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)

        # Project each scale to a common embed dim.
        self.lateral = nn.ModuleList([
            _conv_bn_act(c, embed_dim, k=1) for c in in_channels
        ])

        # Lane-aware aggregator: top-down add then a small refinement conv.
        self.refine = _conv_bn_act(embed_dim, embed_dim, k=3)

        # Fixed prior anchor embeddings (max_lanes priors).
        self.lane_query = nn.Parameter(torch.randn(max_lanes, embed_dim) * 0.02)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_points * 2),
        )
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)

        if self.mask_aux:
            self.mask_decoder = nn.Sequential(
                _conv_bn_act(embed_dim, embed_dim // 2, k=3),
                nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
            )

    def _topdown(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats are ordered low-res -> high-res. Reduce to a single
        # high-resolution tensor by progressive upsample-add.
        proj = [layer(f) for layer, f in zip(self.lateral, feats)]
        # Sort by ascending H*W -> low-res to high-res.
        order = sorted(range(len(proj)), key=lambda i: proj[i].shape[-2] * proj[i].shape[-1])
        merged = proj[order[0]]
        for k in order[1:]:
            target = proj[k]
            merged = F.interpolate(merged, size=target.shape[-2:], mode='bilinear', align_corners=False) + target
        merged = self.refine(merged)
        return merged

    def forward(self, feats: List[torch.Tensor]) -> dict:
        assert isinstance(feats, (list, tuple)) and len(feats) >= 1, 'feats must be a non-empty list'
        x = self._topdown(list(feats))
        B, C, H, W = x.shape

        # Pool to a per-image embedding then broadcast against learned lane queries.
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)        # (B, C)
        # Expand: per-image, per-lane embedding = lane_query + image global feature.
        q = self.lane_query.unsqueeze(0).expand(B, -1, -1)  # (B, L, C)
        img = gap.unsqueeze(1).expand_as(q)                 # (B, L, C)
        per_lane = q + img                                  # (B, L, C)

        cls_logits = self.cls_head(per_lane).squeeze(-1)    # (B, L)
        coord = self.reg_head(per_lane).reshape(B, self.max_lanes, self.num_points, 2)
        coord_pred = torch.sigmoid(coord)                   # constrain to [0, 1]
        lane_class_logits = self.lane_class_head(per_lane)  # (B, L, num_lane_classes)

        out = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_class_logits': lane_class_logits,
        }
        if self.mask_aux:
            mask = self.mask_decoder(x)
            mask = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
            out['mask_logit'] = mask
        return out
