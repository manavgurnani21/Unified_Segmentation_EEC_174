"""Phase B5: Bezier-aware loss components.

Drop-in replacements for CLRKDNet's lane_losses (P5 polyline):
    FocalLossForLane (binary prior classifier)  - REUSED from P5 unchanged
    lane_bezier_geom_loss                       - REPLACES xytl smooth_l1
    bezier_line_iou / bezier_liou_loss          - REPLACES LineIoU on x-grid
    bezier_distance_cost                        - REPLACES distance_cost
    complexity_penalty_loss                     - NEW (LCM only)
    assign_bezier                               - REPLACES P5's dynamic assign

All loss-side defenses learned during the polyline P8 debug stay in place:
    - geometric L1 diff is clamped to +/- normalized-pixel `diff_clamp`
      so a stage-1 out-of-range prediction can't produce a 6000+ spike
      (mirrors the +/-100 px clamp added in iter-1 of debug_record.md)
    - degenerate empty-target images are handled explicitly
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_migration_root() -> Path:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            return cur
        cur = cur.parent
    for ancestor in here.parents:
        cand = ancestor / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration'
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate rmt_ppad_migration/ from {here}")


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


_MIG = _find_migration_root()
_bops = _import_by_path(
    'b5_bezier_ops',
    _MIG / 'extensions' / 'bezier_lcm' / 'Bops' / 'tools' / 'bezier_ops.py',
)
_p5 = _import_by_path(
    'b5_p5_lane_losses',
    _MIG / 'P5_loss' / 'tools' / 'lane_losses.py',
)
FocalLossForLane = _p5.FocalLossForLane  # type: ignore[attr-defined]
render_with_validity = _bops.render_with_validity  # type: ignore[attr-defined]
render_cubic = _bops.render_cubic  # type: ignore[attr-defined]
render_mixture = _bops.render_mixture  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Geometric loss on control points + validity
# ---------------------------------------------------------------------------

def lane_bezier_geom_loss(
    pred_bez: torch.Tensor,
    target_bez: torch.Tensor,
    diff_clamp: float = 1.5,
    cp_weight: float = 1.0,
    validity_weight: float = 0.5,
) -> torch.Tensor:
    """Smooth-L1 on Bezier control points (+ validity range).

    Args:
        pred_bez, target_bez: (N, 16) Bezier vectors.
        diff_clamp: clamp the per-element diff at +/- diff_clamp (in
            normalized [0, 1] space). Mirrors the +/-100 px clamp added
            to the polyline xytl loss in debug_record.md iter-1: prevents
            an out-of-range stage-1 prediction from contributing
            thousands to the loss before the model has had a chance to
            learn. Control points live in [-0.2, 1.2] so a diff of
            magnitude 1.5 covers the legal range; outliers get gradient
            clipped to zero.
        cp_weight: weight on the 8 control-point diffs.
        validity_weight: weight on (t_start, t_end) diffs.
    Returns:
        scalar loss.
    """
    if pred_bez.shape[0] == 0:
        return pred_bez.new_zeros(())

    # Control point part (indices 2:10)
    pred_cp = pred_bez[:, 2:10]
    targ_cp = target_bez[:, 2:10]
    diff_cp = (pred_cp - targ_cp).clamp(min=-diff_clamp, max=diff_clamp)
    cp_loss = F.smooth_l1_loss(
        diff_cp, torch.zeros_like(diff_cp), reduction='none',
    ).mean()

    # Validity range (indices 10, 11)
    pred_val = pred_bez[:, 10:12]
    targ_val = target_bez[:, 10:12]
    diff_val = (pred_val - targ_val).clamp(min=-1.0, max=1.0)
    val_loss = F.smooth_l1_loss(
        diff_val, torch.zeros_like(diff_val), reduction='none',
    ).mean()

    return cp_weight * cp_loss + validity_weight * val_loss


# ---------------------------------------------------------------------------
# Bezier LineIoU (renderable-curve version of CLRKDNet's LineIoU)
# ---------------------------------------------------------------------------

def _render_xs(
    bez: torch.Tensor,
    img_w: int,
    num_samples: int,
    degree_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Sample Bezier x-coords at uniform t in [t_start, t_end].

    Returns pixel-space xs (N, num_samples). If degree_weights is given,
    a 3-way mixture is used (LCM); otherwise pure cubic.
    """
    cp8 = bez[:, 2:10]
    t_start = bez[:, 10]
    t_end = bez[:, 11]
    mode = "mixture" if degree_weights is not None else "cubic"
    xs, _ys = render_with_validity(
        cp8=cp8, t_start=t_start, t_end=t_end,
        n_samples=num_samples, mode=mode, degree_weights=degree_weights,
    )
    return xs * (img_w - 1)  # back to pixel coords


def bezier_line_iou(
    pred_bez: torch.Tensor,
    target_bez: torch.Tensor,
    img_w: int = 640,
    num_samples: int = 72,
    line_width: int = 15,
    aligned: bool = True,
    pred_degree_weights: torch.Tensor = None,
    target_degree_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute Line-IoU between two Bezier sets after rendering.

    Mirrors CLRKDNet's `line_iou`, just with curves rendered from
    `render_with_validity` instead of read off a fixed-y grid.
    """
    pred_xs = _render_xs(pred_bez, img_w, num_samples, pred_degree_weights)
    targ_xs = _render_xs(target_bez, img_w, num_samples, target_degree_weights)

    px1 = pred_xs - line_width; px2 = pred_xs + line_width
    tx1 = targ_xs - line_width; tx2 = targ_xs + line_width

    if aligned:
        ovr = torch.minimum(px2, tx2) - torch.maximum(px1, tx1)
        union = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
        invalid_mask = (targ_xs < 0) | (targ_xs >= img_w)
    else:
        n_pred = pred_xs.shape[0]
        ovr = torch.minimum(px2[:, None, :], tx2[None, ...]) \
            - torch.maximum(px1[:, None, :], tx1[None, ...])
        union = torch.maximum(px2[:, None, :], tx2[None, ...]) \
            - torch.minimum(px1[:, None, :], tx1[None, ...])
        invalid_mask = (targ_xs[None, ...] < 0) | (targ_xs[None, ...] >= img_w)
        invalid_mask = invalid_mask.expand(n_pred, -1, -1)

    ovr = ovr.masked_fill(invalid_mask, 0.0)
    union = union.masked_fill(invalid_mask, 0.0)
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def bezier_liou_loss(
    pred_bez: torch.Tensor,
    target_bez: torch.Tensor,
    img_w: int = 640,
    num_samples: int = 72,
    line_width: int = 15,
    pred_degree_weights: torch.Tensor = None,
) -> torch.Tensor:
    """1 - mean line-IoU. The training-side IoU loss."""
    if pred_bez.shape[0] == 0:
        return pred_bez.new_zeros(())
    iou = bezier_line_iou(
        pred_bez, target_bez,
        img_w=img_w, num_samples=num_samples, line_width=line_width,
        aligned=True,
        pred_degree_weights=pred_degree_weights,
    )
    return (1.0 - iou).mean()


# ---------------------------------------------------------------------------
# Pairwise costs for dynamic-k SimOTA assign
# ---------------------------------------------------------------------------

def bezier_distance_cost(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    img_w: int = 640,
    num_samples: int = 36,
) -> torch.Tensor:
    """Pair-wise mean-pixel distance between prior Beziers and target Beziers.

    Returns: (num_priors, num_targets) tensor.
    """
    pred_xs = _render_xs(predictions, img_w, num_samples)  # (P, S)
    targ_xs = _render_xs(targets, img_w, num_samples)      # (T, S)
    # Distance in x only (mirrors CLRKDNet's distance_cost which is also
    # x-only - the y grid is implicit). Pairwise.
    diff = (pred_xs[:, None, :] - targ_xs[None, :, :]).abs()  # (P, T, S)
    # Ignore the parts of the target curve that lie outside the image
    invalid = (targ_xs < 0) | (targ_xs >= img_w)              # (T, S)
    diff = diff.masked_fill(invalid[None, :, :], 0.0)
    lengths = (~invalid).sum(dim=1).clamp(min=1).to(diff.dtype)  # (T,)
    dist = diff.sum(dim=-1) / lengths[None, :]
    return dist


def focal_cost(
    cls_pred: torch.Tensor, gt_labels: torch.Tensor,
    alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-12,
) -> torch.Tensor:
    """Same as the P5 focal_cost - duplicated locally so this module is
    self-contained and we don't reach into P5 internals."""
    pt = F.softmax(cls_pred, dim=1)
    pt = pt.clamp(eps, 1 - eps)
    pt_pos = pt[:, 1:2]  # (P, 1)
    pt_neg = pt[:, 0:1]
    cost_pos = -alpha * ((1 - pt_pos) ** gamma) * pt_pos.log()
    cost_neg = -(1 - alpha) * (pt_neg ** gamma) * (1 - pt_pos).log()
    cost = (cost_pos - cost_neg).expand(-1, gt_labels.shape[0])
    return cost


def dynamic_k_assign_bezier(
    cost: torch.Tensor, pair_wise_ious: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k Hungarian-style assign, dynamic k per target. Mirrors CLRKDNet."""
    num_priors, num_gts = cost.shape
    matching_matrix = torch.zeros_like(cost)
    # dynamic k per gt: number of "candidate priors" (top-N priors by iou)
    candidate_k = max(1, min(num_priors, 4))
    topk_ious, _ = torch.topk(pair_wise_ious, candidate_k, dim=0)  # (k, T)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)        # (T,)
    for i in range(num_gts):
        _, pos_idx = torch.topk(cost[:, i], k=dynamic_ks[i].item(), largest=False)
        matching_matrix[pos_idx, i] = 1.0
    # Resolve "one prior matched to multiple gts" - keep the lower cost
    prior_match_gt_more_than_one = matching_matrix.sum(1) > 1
    if prior_match_gt_more_than_one.any():
        cost_min, cost_argmin = cost[prior_match_gt_more_than_one].min(dim=1)
        matching_matrix[prior_match_gt_more_than_one] *= 0.0
        matching_matrix[prior_match_gt_more_than_one, cost_argmin] = 1.0
    matched_row = matching_matrix.sum(1) > 0
    matched_col = matching_matrix[matched_row].argmax(1)
    return matched_row.nonzero(as_tuple=False).flatten(), matched_col


def hungarian_assign_bezier(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-to-1 Hungarian matching for Bezier priors. See lane_losses.
    hungarian_assign for the full rationale - dynamic-k's batch-flickering
    positive labels collapse the cls to a uniform sigmoid; Hungarian's
    deterministic 1-prior-per-GT labels let it separate.
    """
    from scipy.optimize import linear_sum_assignment
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    device = cost.device
    prior_idx = torch.as_tensor(row_ind, dtype=torch.long, device=device)
    gt_idx = torch.as_tensor(col_ind, dtype=torch.long, device=device)
    return prior_idx, gt_idx


def assign_bezier(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    img_w: int = 640,
    num_samples: int = 36,
    distance_weight: float = 3.0,
    focal_weight: float = 1.0,
    iou_weight: float = 2.0,
    match: str = 'hungarian',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign for Bezier priors.

    predictions: (P, 16). Slots [0, 1] are cls logits; [2:10] are CPs;
        [10, 11] are validity; rest are LCM / reserved.
    targets:     (T, 16)
    match: 'hungarian' (1-to-1, deterministic; DEFAULT - fixes cls
        collapse) or 'dynamic_k' (SimOTA top-k; original, for ablation).
    Returns:
        matched_row_inds: (M,) prior indices
        matched_col_inds: (M,) target indices

    Cost combines (a) pair-wise distance of rendered curves, (b) per-prior
    focal cost, (c) negative IoU.
    """
    num_priors = predictions.shape[0]
    num_gts = targets.shape[0]
    if num_gts == 0:
        empty = predictions.new_zeros(0, dtype=torch.long)
        return empty, empty.clone()

    dist_cost = bezier_distance_cost(
        predictions, targets, img_w=img_w, num_samples=num_samples,
    )
    cls_pred = predictions[:, :2]
    gt_labels = targets.new_ones(num_gts, dtype=torch.long)
    fcost = focal_cost(cls_pred, gt_labels)  # (P, T)

    pair_iou = bezier_line_iou(
        predictions, targets, img_w=img_w, num_samples=num_samples,
        line_width=15, aligned=False,
    )  # (P, T)

    total_cost = (
        distance_weight * dist_cost
        + focal_weight * fcost
        + iou_weight * (1 - pair_iou)
    )
    if match == 'hungarian':
        return hungarian_assign_bezier(total_cost)
    return dynamic_k_assign_bezier(total_cost, pair_iou)


# ---------------------------------------------------------------------------
# LCM complexity penalty
# ---------------------------------------------------------------------------

def complexity_penalty_loss(
    degree_weights: torch.Tensor,
    matched_mask: torch.Tensor = None,
    lam_2: float = 1.0, lam_3: float = 2.0,
) -> torch.Tensor:
    """Occam's-razor penalty over LCM degree weights.

    Args:
        degree_weights: (B, P, 3)
        matched_mask:   optional (B, P) bool mask. Apply penalty only on
                        priors that matched a real lane. If None, all
                        priors contribute (incl. negatives).
    Returns:
        scalar mean(w_2 * lam_2 + w_3 * lam_3) over matched priors.
    """
    penalties = degree_weights[..., 1] * lam_2 + degree_weights[..., 2] * lam_3
    if matched_mask is None:
        return penalties.mean()
    if matched_mask.sum() == 0:
        return penalties.new_zeros(())
    return penalties[matched_mask].mean()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    torch.manual_seed(0)
    # Two near-identical lanes
    pred = torch.zeros(4, 16)
    targ = torch.zeros(4, 16)
    # Set pos score and 4 control points on a diagonal-ish line
    pred[:, 1] = 1.0; targ[:, 1] = 1.0
    targ[:, 2:6] = torch.tensor([0.10, 0.30, 0.50, 0.70])  # P0..3 x
    targ[:, 6:10] = torch.tensor([0.90, 0.70, 0.30, 0.10])  # P0..3 y
    targ[:, 10] = 0.0; targ[:, 11] = 1.0
    pred[:, 2:10] = targ[:, 2:10] + 0.01 * torch.randn(4, 8)
    pred[:, 10] = 0.0; pred[:, 11] = 1.0

    g = lane_bezier_geom_loss(pred, targ)
    print(f'[B5.smoke] geom_loss (close pred) = {g.item():.5f}')
    assert g.item() < 0.01

    iou = bezier_liou_loss(pred, targ, img_w=640)
    print(f'[B5.smoke] liou_loss (close pred) = {iou.item():.5f}')
    assert iou.item() < 0.05

    # Pathological pred: control points way out of range
    bad = pred.clone()
    bad[:, 2:10] += 10.0
    g_bad = lane_bezier_geom_loss(bad, targ)
    print(f'[B5.smoke] geom_loss (far pred)   = {g_bad.item():.5f}')
    # With diff_clamp=1.5, per-element max smooth_l1 = 1.0, so cp_loss <= 1.0
    assert g_bad.item() < 2.0, 'diff_clamp should bound the geom loss'

    # Assign on a fresh prior set
    priors = torch.zeros(16, 16)
    priors[:, 1] = 0.5  # initial cls logit
    priors[:, 2:10] = 0.5 + 0.2 * torch.randn(16, 8)
    priors[:, 10] = 0.0; priors[:, 11] = 1.0
    matched_row, matched_col = assign_bezier(priors, targ)
    print(f'[B5.smoke] matched {len(matched_row)} priors to {targ.shape[0]} gts')
    assert len(matched_row) >= 1, 'should match at least one prior'

    # Complexity penalty
    w = torch.tensor([[0.1, 0.3, 0.6], [0.05, 0.05, 0.90]])
    p = complexity_penalty_loss(w[None, ...])
    print(f'[B5.smoke] complexity_penalty = {p.item():.5f}')
    assert p.item() > 0.5

    print('[B5.smoke] OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(_smoke_test())


__all__ = [
    'FocalLossForLane', 'lane_bezier_geom_loss',
    'bezier_line_iou', 'bezier_liou_loss',
    'bezier_distance_cost', 'assign_bezier',
    'complexity_penalty_loss',
]
