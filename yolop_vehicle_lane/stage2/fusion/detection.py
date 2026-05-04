from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DetectionLossConfig:
    cls_loss_weight: float = 1.0
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    obj_loss_weight: float = 1.0


class SimpleVehicleDetectionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, num_classes: int = 1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.obj_head = nn.Conv2d(hidden_dim, 1, 1)
        self.box_head = nn.Conv2d(hidden_dim, 4, 1)
        self.cls_head = nn.Conv2d(hidden_dim, self.num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = feats[0]
        x = self.stem(x)
        return {
            'obj_logits': self.obj_head(x),
            'box_raw': self.box_head(x),
            'cls_logits': self.cls_head(x),
        }


def read_yolo_label(path: Path) -> torch.Tensor:
    rows: List[List[float]] = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    a = cxcywh_to_xyxy(box_a)
    b = cxcywh_to_xyxy(box_b)
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-7
    iou = inter / union
    cx1 = torch.minimum(a[..., 0], b[..., 0])
    cy1 = torch.minimum(a[..., 1], b[..., 1])
    cx2 = torch.maximum(a[..., 2], b[..., 2])
    cy2 = torch.maximum(a[..., 3], b[..., 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7
    return iou - (c_area - union) / c_area


class SimpleVehicleDetectionLoss(nn.Module):
    def __init__(self, cfg: Optional[DetectionLossConfig] = None):
        super().__init__()
        self.cfg = cfg or DetectionLossConfig()

    def forward(self, pred: Dict[str, torch.Tensor], targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        obj_logits = pred['obj_logits']
        box_raw = pred['box_raw']
        cls_logits = pred['cls_logits']
        bsz, _one, h, w = obj_logits.shape
        device = obj_logits.device

        obj_target = torch.zeros((bsz, 1, h, w), device=device)
        box_target = torch.zeros((bsz, 4, h, w), device=device)
        cls_target = torch.zeros((bsz, cls_logits.shape[1], h, w), device=device)
        positive = torch.zeros((bsz, 1, h, w), device=device)

        for b, labels in enumerate(targets):
            labels = labels.to(device)
            if labels.numel() == 0:
                continue
            labels = labels[(labels[:, 1:5] >= 0).all(dim=1) & (labels[:, 1:5] <= 1).all(dim=1)]
            if labels.numel() == 0:
                continue
            cx = labels[:, 1].clamp(0, 0.9999)
            cy = labels[:, 2].clamp(0, 0.9999)
            gx = torch.floor(cx * w).long().clamp(0, w - 1)
            gy = torch.floor(cy * h).long().clamp(0, h - 1)
            for i in range(labels.shape[0]):
                obj_target[b, 0, gy[i], gx[i]] = 1.0
                positive[b, 0, gy[i], gx[i]] = 1.0
                box_target[b, :, gy[i], gx[i]] = labels[i, 1:5]
                class_id = int(labels[i, 0].item())
                if 0 <= class_id < cls_logits.shape[1]:
                    cls_target[b, class_id, gy[i], gx[i]] = 1.0

        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)
        box_pred = torch.sigmoid(box_raw)
        pos_mask = positive.expand_as(box_pred) > 0.5
        if pos_mask.any():
            pred_pos = box_pred.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            tgt_pos = box_target.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            bbox_loss = F.smooth_l1_loss(pred_pos, tgt_pos, beta=0.05)
            giou_loss = (1.0 - generalized_iou(pred_pos, tgt_pos)).mean()
            cls_pos_logits = cls_logits.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            cls_pos_target = cls_target.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            cls_loss = F.binary_cross_entropy_with_logits(cls_pos_logits, cls_pos_target)
        else:
            bbox_loss = obj_logits.new_zeros(())
            giou_loss = obj_logits.new_zeros(())
            cls_loss = obj_logits.new_zeros(())

        total = (
            cfg.obj_loss_weight * obj_loss
            + cfg.cls_loss_weight * cls_loss
            + cfg.bbox_loss_weight * bbox_loss
            + cfg.giou_loss_weight * giou_loss
        )
        return total, {
            'det/obj': obj_loss.detach(),
            'det/cls': cls_loss.detach(),
            'det/bbox': bbox_loss.detach(),
            'det/giou': giou_loss.detach(),
            'det/total': total.detach(),
            'det/positives': positive.sum().detach(),
        }
