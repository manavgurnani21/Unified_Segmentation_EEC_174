"""Phase P5: lane-loss utilities ported from CLRKDNet.

Source files (verbatim ports, kept self-contained so we don't depend on the
CLRKDNet package being installed/compiled at training time):

  - clrkd/models/losses/lineiou_loss.py    -> line_iou, liou_loss
  - clrkd/models/utils/dynamic_assign.py   -> distance_cost, focal_cost,
                                              dynamic_k_assign, assign
  - clrkd/models/losses/focal_loss.py      -> FocalLossForLane (simplified
                                              binary variant per appendix-path3
                                              sec 8.3)

The originals depend on `from clrkd.models.losses.lineiou_loss import line_iou`
relative imports, and FocalLoss in particular has a much heavier general-case
implementation (with one_hot, etc). We only need the 2-class case for CLRHead's
prior classification, so we use the simpler FocalLossForLane that matches the
appendix's spec.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Line IoU (lineiou_loss.py)
# ---------------------------------------------------------------------------

def line_iou(pred: torch.Tensor, target: torch.Tensor, img_w: int,
             length: int = 15, aligned: bool = True) -> torch.Tensor:
    """Calculate the line IoU between predictions and targets.

    Args:
        pred: lane predictions, shape (num_pred, num_points).
        target: ground truth, shape (num_target, num_points).
        img_w: image width.
        length: extended radius (in pixels) for the "line thickness".
        aligned: True for elementwise IoU (loss); False for pair-wise IoUs
            (assignment).
    Returns:
        aligned=True  -> (N,) IoU per row.
        aligned=False -> (num_pred, num_target) IoU matrix.
    """
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...])
               - torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...])
                 - torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.0
    union[invalid_masks] = 0.0
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred: torch.Tensor, target: torch.Tensor, img_w: int,
              length: int = 15) -> torch.Tensor:
    """Mean (1 - line_iou). Use this in the regression branch."""
    return (1 - line_iou(pred, target, img_w, length)).mean()


# ---------------------------------------------------------------------------
# Sprint-2 S2.A: CLRerNet angle-aware LaneIoU (WACV'24).
# Ported from external_repos/CLRerNet-main/libs/models/losses/iou_loss.py.
# Key difference vs line_iou: the virtual lane half-width is NOT constant; it
# scales with the LOCAL SLOPE  w = base * sqrt(dx^2 + dy^2)/dy.  Where the lane
# is steep (near-field, x changes fast per row -- our jaggedness zone) the band
# widens, so the IoU penalizes shape error proportional to the local angle.
# line_iou's fixed band under-penalizes exactly there.  Same pixel-coord I/O as
# line_iou so it is a drop-in (gated by LANE_IOU_TYPE=laneiou).
# ---------------------------------------------------------------------------

def _lane_iou_widths(pred_px: torch.Tensor, target_px: torch.Tensor,
                     img_w: int, img_h: int, base_width: float):
    """Per-row slope-scaled half-widths (in PIXELS) for pred and target.

    pred_px/target_px: (..., Nr) x-grids in pixels. base_width is the half-width
    in pixels at zero slope (== the old constant `length`). Returns two tensors
    broadcastable to the x-grid.
    """
    n_strips = pred_px.shape[-1] - 1
    dy = img_h / n_strips * 2.0                     # span of two row-grids (px)

    def _w(x_px):
        # central x-difference over two rows (px); pad both ends to keep width.
        dx = x_px[..., 2:] - x_px[..., :-2]
        w = base_width * torch.sqrt(dx * dx + dy * dy) / dy
        return torch.cat([w[..., :1], w, w[..., -1:]], dim=-1)

    return _w(pred_px.detach()), _w(target_px)


def lane_iou(pred: torch.Tensor, target: torch.Tensor, img_w: int,
             length: int = 15, aligned: bool = True,
             img_h: int = 640, use_giou: bool = True) -> torch.Tensor:
    """Angle-aware LaneIoU (CLRerNet). Same shape contract as line_iou:
    aligned=True -> (N,); aligned=False -> (num_pred, num_target).

    pred/target are x-grids in PIXELS. `length` is the zero-slope half-width.

    use_giou (CLRerNet `use_giou`, default ON): on rows where the pred and GT
    bands do NOT overlap (ovr < 0), add the disjoint span to the union (the
    "virtual union" of CLRerNet's _set_invalid_*). This makes the IoU DECREASE
    smoothly as two separated lanes move apart, so a far-off prior still gets a
    gradient pulling it toward its GT -- the mechanism that fights the
    center-collapse (startx_std ~ 0.04). Plain IoU clamps disjoint rows to 0 and
    gives no such gradient.
    """
    if aligned:
        pw, tw = _lane_iou_widths(pred, target, img_w, img_h, length)
        px1, px2 = pred - pw, pred + pw
        tx1, tx2 = target - tw, target + tw
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        # pairwise: widths computed per-lane, then broadcast across the cross.
        pw, _ = _lane_iou_widths(pred, pred, img_w, img_h, length)      # (P, Nr)
        _, tw = _lane_iou_widths(target, target, img_w, img_h, length)  # (T, Nr)
        px1 = (pred - pw)[:, None, :]
        px2 = (pred + pw)[:, None, :]
        tx1 = (target - tw)[None, :, :]
        tx2 = (target + tw)[None, :, :]
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

    if use_giou:
        # On separated rows ovr<0; clamp the overlap to 0 (no negative overlap)
        # but KEEP the (now larger) union, so IoU = 0/union shrinks toward 0 as
        # the gap grows -- a smooth, differentiable "how far apart" signal. The
        # union already equals max-min span, which grows with the gap, so simply
        # not zeroing it (only zeroing the negative overlap) realizes the
        # virtual-union effect in this aligned/grid formulation.
        ovr = ovr.clamp(min=0.0)

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr = ovr.masked_fill(invalid_masks, 0.0)
    union = union.masked_fill(invalid_masks, 0.0)
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def lane_iou_loss(pred: torch.Tensor, target: torch.Tensor, img_w: int,
                  length: int = 15, img_h: int = 640,
                  use_giou: bool = True) -> torch.Tensor:
    """Mean (1 - lane_iou). Drop-in for liou_loss with angle-aware widths."""
    return (1 - lane_iou(pred, target, img_w, length, aligned=True,
                         img_h=img_h, use_giou=use_giou)).mean()


# ---------------------------------------------------------------------------
# Dynamic assign (dynamic_assign.py)
# ---------------------------------------------------------------------------

def distance_cost(predictions: torch.Tensor, targets: torch.Tensor,
                  img_w: int) -> torch.Tensor:
    """Pair-wise abs-distance cost between every (prior, target) pair,
    averaged over the in-image x-offsets."""
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    predictions = torch.repeat_interleave(
        predictions, num_targets, dim=0,
    )[..., 6:]
    targets = torch.cat(num_priors * [targets])[..., 6:]

    invalid_masks = (targets < 0) | (targets >= img_w)
    lengths = (~invalid_masks).sum(dim=1)
    distances = torch.abs(targets - predictions)
    distances[invalid_masks] = 0.0
    distances = distances.sum(dim=1) / (lengths.float() + 1e-9)
    return distances.view(num_priors, num_targets)


def focal_cost(cls_pred: torch.Tensor, gt_labels: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0,
               eps: float = 1e-12) -> torch.Tensor:
    """Pair-wise classification cost (focal form). cls_pred is (N, C);
    gt_labels is (num_gt,) of class indices into C."""
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]


def dynamic_k_assign(cost: torch.Tensor,
                     pair_wise_ious: torch.Tensor) -> tuple:
    """SimOTA-style dynamic top-k matching. Returns (prior_idx, gt_idx)."""
    matching_matrix = torch.zeros_like(cost)
    ious_matrix = pair_wise_ious
    ious_matrix[ious_matrix < 0] = 0.0
    n_candidate_k = 4
    topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    num_gt = cost.shape[1]
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(
            cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False,
        )
        matching_matrix[pos_idx, gt_idx] = 1.0

    matched_gt = matching_matrix.sum(1)
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
        matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_gt > 1, cost_argmin] = 1.0

    prior_idx = matching_matrix.sum(1).nonzero()
    gt_idx = matching_matrix[prior_idx].argmax(-1)
    return prior_idx.flatten(), gt_idx.flatten()


def hungarian_assign(cost: torch.Tensor) -> tuple:
    """1-to-1 Hungarian matching. Returns (prior_idx, gt_idx).

    Lower cost = better match (same sign convention as dynamic_k_assign,
    which selects with largest=False). `scipy.linear_sum_assignment`
    minimizes total cost, so the raw cost matrix is passed as-is.

    On a (num_priors=192, num_gt=K) matrix this returns exactly K matches
    - ONE prior per GT, chosen globally optimally and DETERMINISTICALLY.
    That determinism is the whole point: dynamic-k labels the same prior
    positive in some batches / negative in others, so the binary cls
    converges to a uniform 0.5 sigmoid (the only stable equilibrium) and
    the decoded lane IoU freezes. Hungarian gives every matched prior a
    stable positive label across batches, so the cls can actually
    separate pos from neg (Exp2RR/NB47 broke an 11-experiment plateau
    exactly this way: pos-neg gap 0.01 -> 0.099).
    """
    from scipy.optimize import linear_sum_assignment
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    device = cost.device
    prior_idx = torch.as_tensor(row_ind, dtype=torch.long, device=device)
    gt_idx = torch.as_tensor(col_ind, dtype=torch.long, device=device)
    return prior_idx, gt_idx


def assign(predictions: torch.Tensor, targets: torch.Tensor,
           img_w: int, img_h: int,
           distance_cost_weight: float = 3.0,
           cls_cost_weight: float = 1.0,
           match: str = 'hungarian') -> tuple:
    """Match priors to ground-truth lanes on a (distance+xy+theta+cls) cost.

    Args:
        predictions: (num_priors, 78) - the model's per-prior output, normalized.
        targets:     (num_targets, 78) - per-image GT lanes from build_full_target_tensor.
        img_w, img_h: target image size (640 in our setup).
        match: 'hungarian' (1-to-1, deterministic; DEFAULT - fixes cls
            collapse) or 'dynamic_k' (SimOTA top-k; the original CLRKDNet
            scheme, kept for ablation/revert).
    Returns:
        (matched_row_inds, matched_col_inds) - which priors match which GTs.
    """
    predictions = predictions.detach().clone()
    predictions[:, 3] *= (img_w - 1)
    predictions[:, 6:] *= (img_w - 1)
    targets = targets.detach().clone()

    distances_score = distance_cost(predictions, targets, img_w)
    distances_score = 1 - (distances_score / torch.max(distances_score)) + 1e-2

    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]
    target_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)

    start_xys_score = torch.cdist(
        prediction_start_xys, target_start_xys, p=2,
    ).reshape(num_priors, num_targets)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(
        predictions[:, 4].unsqueeze(-1), target_thetas, p=1,
    ).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = (-(distances_score * start_xys_score * theta_score) ** 2
            * distance_cost_weight + cls_score * cls_cost_weight)

    # S2.A (optional): add an angle-aware LaneIoU term to the matcher cost so the
    # assignment, not just the loss, is geometry-faithful. Env LANE_IOU_TYPE=
    # laneiou turns it on; default leaves the hungarian cost unchanged. Lower IoU
    # -> higher cost (we SUBTRACT a positive-IoU reward weighted by iou_cost_w).
    import os as _os
    # The MATCHER's use of LaneIoU is gated separately from the LOSS, so the
    # ablation can isolate loss-only / matcher-only / both. LANE_IOU_MATCH
    # defaults to follow LANE_IOU_TYPE (back-compat: 'laneiou' enables both),
    # but can be set explicitly to 'line'/'laneiou' to decouple them.
    _match_iou = _os.environ.get(
        'LANE_IOU_MATCH', _os.environ.get('LANE_IOU_TYPE', 'line')).strip().lower()
    if _match_iou == 'laneiou':
        _iou_w = float(_os.environ.get('LANE_IOU_COST_W', '2.0') or 2.0)
        # Two-width scheme (CLRerNet): the matcher cost uses a WIDER band than
        # the loss so the assignment is more tolerant of coarse early geometry
        # (CLRerNet: 30/800 cost vs 7.5/800 dynamic-k). In our 640-px frame the
        # default cost width = 30/800*640 = 24 px. Env LANE_IOU_COST_WIDTH.
        _cost_width = float(_os.environ.get('LANE_IOU_COST_WIDTH', '24') or 24)
        liou = lane_iou(predictions[..., 6:], targets[..., 6:], img_w,
                        length=_cost_width, aligned=False, img_h=img_h,
                        use_giou=True)                        # (P, T) in [0,1]
        cost = cost - _iou_w * liou

    if match == 'hungarian':
        return hungarian_assign(cost)
    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)
    return dynamic_k_assign(cost, iou)


# ---------------------------------------------------------------------------
# Focal loss (focal_loss.py) - simplified binary version per appendix sec 8.3
# ---------------------------------------------------------------------------

class FocalLossForLane(nn.Module):
    """Per-prior focal loss for the 2-class (lane vs bg) prior classifier.

    Returns shape (N,) so caller can do `.sum() / num_gt` per CLRHead's loss.
    Equivalent (for the binary case) to CLRKDNet's general FocalLoss in
    clrkd/models/losses/focal_loss.py without the one_hot overhead.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pt = F.softmax(pred, dim=1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(min=1e-8, max=1 - 1e-8)
        return -self.alpha * (1 - pt) ** self.gamma * pt.log()


__all__ = [
    'line_iou', 'liou_loss', 'lane_iou', 'lane_iou_loss',
    'distance_cost', 'focal_cost', 'dynamic_k_assign', 'hungarian_assign',
    'assign', 'FocalLossForLane',
]


def smoke_test() -> int:
    """Quick sanity test for the ported lane-loss utilities."""
    print('[smoke] lane_losses starting', flush=True)
    torch.manual_seed(0)

    # 1. line_iou: aligned + non-aligned
    pred = torch.tensor([[100.0, 110.0, 120.0]])
    tgt  = torch.tensor([[100.0, 110.0, 120.0]])
    iou_a = line_iou(pred, tgt, img_w=640, length=15, aligned=True)
    print(f'[smoke] line_iou(identical): {iou_a.item():.3f}  (expect 1.000)')
    if abs(iou_a.item() - 1.0) > 1e-3:
        print('[smoke] FAIL: identical IoU != 1')
        return 1

    pred2 = torch.randn(5, 72) * 100 + 320
    tgt2  = torch.randn(3, 72) * 100 + 320
    iou_mat = line_iou(pred2, tgt2, img_w=640, length=15, aligned=False)
    print(f'[smoke] pair-wise IoU shape: {tuple(iou_mat.shape)}  (expect (5, 3))')
    if tuple(iou_mat.shape) != (5, 3):
        print('[smoke] FAIL: non-aligned IoU shape')
        return 1

    # 2. liou_loss is in [0, 1]
    loss = liou_loss(pred, tgt, img_w=640, length=15)
    print(f'[smoke] liou_loss(identical): {loss.item():.4f}  (expect 0)')

    # 3. assign on a synthetic batch
    n_priors = 192
    n_gt = 3
    predictions = torch.randn(n_priors, 78) * 0.1
    predictions[:, :2] = torch.randn(n_priors, 2)  # cls logits
    predictions[:, 2:5] = torch.rand(n_priors, 3)  # start_y, start_x, theta in [0, 1]
    predictions[:, 5] = torch.randint(5, 60, (n_priors,)).float()  # length
    targets = torch.zeros(n_gt, 78)
    targets[:, 1] = 1.0   # pos_score = 1
    targets[:, 2] = torch.tensor([0.1, 0.3, 0.5])  # start_y normalized
    targets[:, 3] = torch.tensor([100.0, 300.0, 500.0])
    targets[:, 4] = torch.tensor([0.3, 0.5, 0.7])
    targets[:, 5] = torch.tensor([40.0, 50.0, 30.0])
    targets[:, 6:] = torch.rand(n_gt, 72) * 640
    matched_row, matched_col = assign(predictions, targets, img_w=640, img_h=640)
    print(f'[smoke] assign: matched {len(matched_row)} priors to {n_gt} GTs')
    if len(matched_row) < 1 or len(matched_row) != len(matched_col):
        print('[smoke] FAIL: assign returned bad shapes')
        return 1

    # 4. FocalLossForLane is finite and differentiable
    cls_pred = torch.randn(192, 2, requires_grad=True)
    cls_target = torch.randint(0, 2, (192,))
    fl = FocalLossForLane()(cls_pred, cls_target)
    print(f'[smoke] FocalLossForLane shape={tuple(fl.shape)} sum={fl.sum().item():.3f}')
    if not torch.isfinite(fl).all():
        print('[smoke] FAIL: focal loss non-finite')
        return 1
    fl.sum().backward()
    if cls_pred.grad is None or not torch.isfinite(cls_pred.grad).all():
        print('[smoke] FAIL: backward pass non-finite')
        return 1

    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
