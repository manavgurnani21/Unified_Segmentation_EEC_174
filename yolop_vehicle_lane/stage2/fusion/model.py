"""Minimal Stage 2 fusion model wrapper.

This is intentionally a thin glue layer between:
  - a feature-producing backbone+neck (CSP / RMT / YOLO26 / etc.)
  - a detection head (YOLOPX or RT-DETR)
  - the in-house CLRKD-style curve lane head

The motivation is that Stage 2 experiments differ mainly in WHICH backbone
and detection head are plugged in. The lane head and loss stay the same
across experiments 3-6, so a single small wrapper makes those comparisons
honest.

This file does NOT bundle a backbone — call sites pass one in. Constructing
the actual backbone is left to the experiment notebook so that vendored
RMT / YOLO26 imports stay scoped to their experiment.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lane_head import CurveLaneHead


class FusionModel(nn.Module):
    """Wrapper that combines a feature backbone with a lane head and a
    detection head.

    The backbone is expected to return a list of feature maps. The
    detection head is expected to be an `nn.Module` whose forward
    accepts that list and returns a detector-specific output (loss
    structure handled by detector-specific code at the call site).

    The lane branch always outputs (cls_logits, coord_pred, mask_logit)
    in the format consumed by `FusionLaneLoss`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        detection_head: Optional[nn.Module] = None,
        lane_head: Optional[nn.Module] = None,
        lane_in_indices: Optional[Sequence[int]] = None,
        max_lanes: int = 10,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_channels = list(feature_channels)
        self.detection_head = detection_head

        # By default, send all feature scales to the lane head.
        if lane_in_indices is None:
            self.lane_in_indices = list(range(len(feature_channels)))
        else:
            self.lane_in_indices = list(lane_in_indices)

        if lane_head is None:
            lane_channels = [self.feature_channels[i] for i in self.lane_in_indices]
            self.lane_head = CurveLaneHead(
                in_channels=lane_channels,
                max_lanes=max_lanes,
                num_points=num_points,
                mask_size=mask_size,
            )
        else:
            self.lane_head = lane_head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        feats = list(feats)

        lane_feats = [feats[i] for i in self.lane_in_indices]
        lane_out = self.lane_head(lane_feats)

        det_out = None
        if self.detection_head is not None:
            det_out = self.detection_head(feats)

        return {
            'lane': lane_out,
            'det': det_out,
            'features': feats,
        }
