"""Stage 2 fusion loss module.

Lane-curve representation used here is the (existence, points, visibility,
lane_type) form produced by `stage2.fusion.lane_targets.frame_to_lane_targets`.
The CLRKDNet (max_lanes, 78) line-prior representation is handled by the
vendored CLRKDNet trainer; this module is the in-house lane head's loss.

Composition (Part E):

    L_lane = w_cls   * L_cls
           + w_reg   * L_reg
           + w_iou   * L_line_iou
           + w_mask  * L_mask_aux
           + w_smooth * L_smooth
           + w_distill * L_distill   (optional)

    L_total = uncertainty_weight(L_det, L_lane)   (optional)
            -- or --
    L_total = L_det + lambda_lane * L_lane        (default baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FusionLossConfig:
    # Top-level multi-task balance.
    lambda_lane: float = 1.0          # used when uncertainty=False
    use_uncertainty: bool = False     # learnable log-variance weights

    # Lane sub-loss weights.
    w_cls: float = 2.0
    w_reg: float = 0.5
    w_iou: float = 2.0
    w_mask: float = 1.0
    w_smooth: float = 0.05
    w_distill: float = 0.0            # off by default

    # Focal loss params for lane existence/classification.
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # LineIoU sampling band (in normalized image-x units).
    line_iou_radius: float = 0.015

    # Smoothness penalty (2nd derivative of x along the curve).
    smooth_eps: float = 1e-6


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _binary_focal_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Binary focal loss on logits."""
    bce = F.binary_cross_entropy_with_logits(logit, target.float(), reduction='none')
    p = torch.sigmoid(logit)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).clamp(min=0).pow(gamma) * bce
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _line_iou_1d(pred_x: torch.Tensor, gt_x: torch.Tensor,
                 mask: torch.Tensor, radius: float) -> torch.Tensor:
    """LineIoU between two per-row x sequences using a band of `radius`.

    pred_x / gt_x : (..., N) normalized x in [0, 1]
    mask          : (..., N) 1 for valid rows, 0 otherwise
    Returns: (...,) IoU in [0, 1]
    """
    pa_lo = pred_x - radius
    pa_hi = pred_x + radius
    pb_lo = gt_x - radius
    pb_hi = gt_x + radius
    inter = (torch.minimum(pa_hi, pb_hi) - torch.maximum(pa_lo, pb_lo)).clamp(min=0)
    union = (torch.maximum(pa_hi, pb_hi) - torch.minimum(pa_lo, pb_lo)).clamp(min=1e-6)
    iou = (inter * mask).sum(dim=-1) / ((union * mask).sum(dim=-1) + 1e-6)
    return iou


def _dice_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum() + eps
    den = (p + target).sum() + eps
    return 1 - num / den


def _smoothness_x(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Second-difference penalty on x-coordinate sequences.

    points : (..., N, 2) normalized
    mask   : (..., N) valid-row indicator
    """
    if points.shape[-2] < 3:
        return points.new_zeros(())
    x = points[..., 0]
    d2 = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    valid = mask[..., 2:] * mask[..., 1:-1] * mask[..., :-2]
    if valid.sum() < 1:
        return points.new_zeros(())
    return ((d2.abs() * valid).sum() / (valid.sum() + 1e-6))


# ---------------------------------------------------------------------------
# Lane fusion loss
# ---------------------------------------------------------------------------
class FusionLaneLoss(nn.Module):
    """Lane-side loss for the in-house curve lane head.

    Predictions (per-image, fixed max_lanes slots):
        cls_logits  : (B, max_lanes)               existence logit
        coord_pred  : (B, max_lanes, num_points, 2) normalized (x, y)
        mask_logit  : (B, 1, H, W) (optional)      auxiliary rendered mask
        cls_logits_dist (optional)                 teacher-style logits for KD
        coord_pred_dist (optional)                 teacher coord for KD

    Targets:
        existence   : (B, max_lanes) {0, 1}
        points      : (B, max_lanes, num_points, 2) normalized
        visibility  : (B, max_lanes, num_points)    {0, 1}
        mask_target : (B, 1, H, W) (optional)
    """

    def __init__(self, cfg: FusionLossConfig = None):
        super().__init__()
        self.cfg = cfg or FusionLossConfig()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        teacher: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg

        cls_logit = pred['cls_logits']                  # (B, L)
        coord_pred = pred['coord_pred']                 # (B, L, N, 2)

        existence = target['existence']                 # (B, L)
        points_gt = target['points']                    # (B, L, N, 2)
        vis = target['visibility']                      # (B, L, N)

        # 1) Existence / classification (focal BCE).
        cls_loss = _binary_focal_loss(
            cls_logit, existence,
            alpha=cfg.focal_alpha, gamma=cfg.focal_gamma,
        )

        # 2) Coordinate regression — only on valid lanes/rows.
        valid_lane = existence.unsqueeze(-1)            # (B, L, 1)
        valid_row = valid_lane * vis                    # (B, L, N)
        diff = (coord_pred - points_gt).abs()           # (B, L, N, 2)
        # SmoothL1 manually, with per-row mask.
        beta = 0.05
        sl1 = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        reg_loss = (sl1.sum(-1) * valid_row).sum() / (valid_row.sum() * 2 + 1e-6)

        # 3) LineIoU on x-coordinates (per lane).
        pred_x = coord_pred[..., 0]
        gt_x = points_gt[..., 0]
        iou = _line_iou_1d(pred_x, gt_x, vis, radius=cfg.line_iou_radius)  # (B, L)
        iou_loss = ((1 - iou) * existence).sum() / (existence.sum() + 1e-6)

        # 4) Smoothness regularization on predicted curves.
        smooth_loss = _smoothness_x(coord_pred, valid_row)

        # 5) Auxiliary mask loss (Dice + BCE).
        if 'mask_logit' in pred and 'mask_target' in target and pred['mask_logit'] is not None:
            mlogit = pred['mask_logit']
            mtarget = target['mask_target']
            bce = F.binary_cross_entropy_with_logits(mlogit, mtarget.float())
            dice = _dice_loss(mlogit, mtarget.float())
            mask_loss = 0.5 * bce + 0.5 * dice
        else:
            mask_loss = cls_logit.new_zeros(())

        # 6) Optional distillation.
        distill_loss = cls_logit.new_zeros(())
        if cfg.w_distill > 0 and teacher is not None:
            t_cls = teacher.get('cls_logits')
            t_coord = teacher.get('coord_pred')
            if t_cls is not None and t_cls.shape == cls_logit.shape:
                distill_loss = distill_loss + F.mse_loss(
                    torch.sigmoid(cls_logit), torch.sigmoid(t_cls).detach()
                )
            if t_coord is not None and t_coord.shape == coord_pred.shape:
                distill_loss = distill_loss + (
                    (coord_pred - t_coord.detach()).abs() * valid_row.unsqueeze(-1)
                ).sum() / (valid_row.sum() * 2 + 1e-6)

        total = (
            cfg.w_cls * cls_loss
            + cfg.w_reg * reg_loss
            + cfg.w_iou * iou_loss
            + cfg.w_mask * mask_loss
            + cfg.w_smooth * smooth_loss
            + cfg.w_distill * distill_loss
        )
        components = {
            'lane/cls': cls_loss.detach(),
            'lane/reg': reg_loss.detach(),
            'lane/line_iou': iou_loss.detach(),
            'lane/mask_aux': mask_loss.detach(),
            'lane/smooth': smooth_loss.detach(),
            'lane/distill': distill_loss.detach(),
            'lane/total': total.detach(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Multi-task uncertainty weighting (Kendall, Gal, Cipolla 2018)
# ---------------------------------------------------------------------------
class UncertaintyMultiTaskLoss(nn.Module):
    """Weighted sum L = sum_i (1 / (2*exp(s_i))) * L_i + s_i / 2 with
    learnable scalars s_i (s_i = log_variance). Uses 1/(2 sigma^2) for
    regression-like losses; ok for cross-entropy too in practice.
    """

    def __init__(self, n_tasks: int = 2, init: float = 0.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.full((n_tasks,), float(init)))

    def forward(self, losses: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = list(losses)
        assert len(losses) == self.log_var.numel(), (
            f'UncertaintyMultiTaskLoss got {len(losses)} losses, expects {self.log_var.numel()}.'
        )
        total = losses[0].new_zeros(())
        comp: Dict[str, torch.Tensor] = {}
        for i, l in enumerate(losses):
            s = self.log_var[i]
            weighted = 0.5 * torch.exp(-s) * l + 0.5 * s
            total = total + weighted
            comp[f'mtl/log_var_{i}'] = s.detach()
            comp[f'mtl/weighted_{i}'] = weighted.detach()
        comp['mtl/total'] = total.detach()
        return total, comp


# ---------------------------------------------------------------------------
# Gradient cosine diagnostic for task-conflict measurement
# ---------------------------------------------------------------------------
def compute_grad_cosine(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    shared_params: Iterable[torch.nn.Parameter],
) -> float:
    """Cosine similarity between gradients of two losses w.r.t. shared params.

    Negative values indicate task conflict on the shared backbone.
    Caller must NOT have called .backward() yet.
    """
    shared_params = [p for p in shared_params if p.requires_grad]
    if not shared_params:
        return float('nan')
    g_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
    g_b = torch.autograd.grad(loss_b, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
    flat_a = []
    flat_b = []
    for ga, gb in zip(g_a, g_b):
        if ga is None or gb is None:
            continue
        flat_a.append(ga.flatten())
        flat_b.append(gb.flatten())
    if not flat_a:
        return float('nan')
    va = torch.cat(flat_a)
    vb = torch.cat(flat_b)
    denom = (va.norm() * vb.norm()).clamp(min=1e-12)
    return float((va @ vb / denom).item())
