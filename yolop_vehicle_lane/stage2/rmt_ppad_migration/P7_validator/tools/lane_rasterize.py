"""Phase P7: rasterize raw CLRHead predictions (B, num_priors, 78) into a
dense mask (B, 1, H, W) that the existing per-pixel IoU/ACC metrics can
consume directly.

Pipeline per prior row:
  1. Softmax across [neg_score, pos_score]; keep scores >= conf_threshold.
  2. Sort surviving rows by pos_score descending, take top max_lanes.
     (Cheap top-k stand-in for CLRKDNet's line-IoU NMS - the latter
     requires the CUDA op which we don't want to force on every Colab
     val run. Acceptable for P7's evaluation because we already cap at
     max_lanes=8 ground-truth lanes per image; any extra duplicates are
     bounded by the same cap before rasterization.)
  3. For each kept prior, use the slot-aligned (start_y, length) interval
     to read x values from row[6+start:6+start+length] and draw a
     polyline on the canvas.

The slot convention exactly matches P1's `build_full_target_tensor_v2` +
P6's `rasterize_lane_target_to_mask`, so a perfect prediction would
produce the same mask shape as the GT target.
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
import torch


def _softmax_pos_score(predictions: torch.Tensor) -> torch.Tensor:
    """predictions[:, :2] are (neg, pos) logits -> return pos-class probability."""
    return torch.softmax(predictions[:, :2], dim=1)[:, 1]


def _select_topn(predictions, scores, conf_threshold, max_lanes):
    """Pick which priors to rasterize.

    Phase 1 change: when `conf_threshold` is None or <= 0, the absolute
    threshold is REMOVED and we keep purely the top-`max_lanes` priors by
    score ranking. The validator now calls with conf_threshold=None.

    Why: with cls collapsed to a uniform ~0.5 sigmoid, every prior clears
    any fixed cutoff equally, so thresholding is a no-op and the top-k tie
    breaks the same way every epoch -> IoU frozen. Pure top-N makes the
    decoded mask depend ONLY on the score RANKING, so the moment the cls
    starts to separate (even within a tight 0.5 cluster) the selected set
    shifts and IoU responds. It turns IoU into a sensitive 'is the cls
    moving at all' instrument.

    Returns (kept_predictions, kept_scores).
    """
    if conf_threshold is None or conf_threshold <= 0:
        kept = predictions
        kept_scores = scores
    else:
        keep_inds = scores >= conf_threshold
        kept = predictions[keep_inds]
        kept_scores = scores[keep_inds]
    if kept.shape[0] > max_lanes:
        top_k = torch.topk(kept_scores, k=max_lanes, largest=True).indices
        kept = kept[top_k]
        kept_scores = kept_scores[top_k]
    return kept, kept_scores


def lane_score_stats(predictions: torch.Tensor, n_bins: int = 10) -> dict:
    """Diagnostic stats over the per-prior pos-scores for one image.

    Returns a dict with min/max/mean/std/spread plus a coarse histogram
    of the pos-class softmax probabilities across `n_bins` bins in
    [0, 1]. Used to answer 'are the 192 priors still tightly clustered at
    0.5 and not moving?'. `spread` (= max - min) and `std` near zero =
    dead/collapsed cls head.
    """
    with torch.no_grad():
        s = _softmax_pos_score(predictions.detach()).float()
        hist = torch.histc(s, bins=n_bins, min=0.0, max=1.0)
        return {
            'min': float(s.min()),
            'max': float(s.max()),
            'mean': float(s.mean()),
            'std': float(s.std()),
            'spread': float(s.max() - s.min()),
            'hist': [int(x) for x in hist.tolist()],
            'n': int(s.numel()),
        }


def _bezier_lanes_to_mask(
    predictions: torch.Tensor,
    img_h: int, img_w: int,
    line_width: int, conf_threshold: float, max_lanes: int,
    n_render: int = 100,
) -> np.ndarray:
    """Bezier-prediction rasterizer (16-D vectors).

    Layout matches B1 / B4:
        row[ 0: 2]  cls logits (neg, pos)
        row[ 2: 6]  P0.x, P1.x, P2.x, P3.x  (normalized)
        row[ 6:10]  P0.y, P1.y, P2.y, P3.y
        row[10]     t_start
        row[11]     t_end
        row[12]     complexity score
        row[13:16]  LCM degree weights (w_1, w_2, w_3) summing to 1 when
                    LCM is active; all zeros when LCM is off.

    If the LCM weights sum to >= 0.99 they're treated as a degree
    mixture and the curve is rendered as `w_1 * K1 + w_2 * K2 + w_3 * K3`
    from the same 4 control points (matching B5's loss-side renderer).
    Otherwise pure cubic.
    """
    import cv2

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    with torch.no_grad():
        scores = _softmax_pos_score(predictions.detach())
        kept, kept_scores = _select_topn(
            predictions, scores, conf_threshold, max_lanes,
        )
        if kept.shape[0] == 0:
            return mask

    arr = kept.detach().cpu().numpy().astype(np.float64)
    ts_uniform = np.linspace(0.0, 1.0, n_render)

    for row in arr:
        # Control points
        cps_x = row[2:6]
        cps_y = row[6:10]
        t_start = float(row[10])
        t_end = float(row[11])
        if t_end - t_start < 0.01:
            continue
        # Map ts_uniform into [t_start, t_end]
        t = t_start + (t_end - t_start) * ts_uniform
        u = 1.0 - t

        # Cubic Bezier eval (always computed; mixture overrides per-K weights)
        x_cubic = (u**3 * cps_x[0]
                   + 3*u**2*t * cps_x[1]
                   + 3*u*t**2 * cps_x[2]
                   + t**3 * cps_x[3])
        y_cubic = (u**3 * cps_y[0]
                   + 3*u**2*t * cps_y[1]
                   + 3*u*t**2 * cps_y[2]
                   + t**3 * cps_y[3])

        deg_weights = row[13:16]
        w_sum = float(deg_weights.sum())
        if w_sum >= 0.99:  # LCM mixture mode
            # K=1: straight line P0 -> P3
            x_lin = u * cps_x[0] + t * cps_x[3]
            y_lin = u * cps_y[0] + t * cps_y[3]
            # K=2: quadratic through midpoint(P1, P2)
            Pmid_x = 0.5 * (cps_x[1] + cps_x[2])
            Pmid_y = 0.5 * (cps_y[1] + cps_y[2])
            x_quad = u**2 * cps_x[0] + 2 * u * t * Pmid_x + t**2 * cps_x[3]
            y_quad = u**2 * cps_y[0] + 2 * u * t * Pmid_y + t**2 * cps_y[3]
            w1, w2, w3 = float(deg_weights[0]), float(deg_weights[1]), float(deg_weights[2])
            xs = w1 * x_lin + w2 * x_quad + w3 * x_cubic
            ys = w1 * y_lin + w2 * y_quad + w3 * y_cubic
        else:
            xs = x_cubic
            ys = y_cubic

        # Normalized -> pixel
        xs_pixel = xs * (img_w - 1)
        ys_pixel = ys * (img_h - 1)
        valid = (xs_pixel >= 0) & (xs_pixel < img_w) & (ys_pixel >= 0) & (ys_pixel < img_h)
        xs_pixel = xs_pixel[valid]
        ys_pixel = ys_pixel[valid]
        if len(xs_pixel) < 2:
            continue
        pts = np.stack([xs_pixel, ys_pixel], axis=-1).round().astype(np.int32)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)],
                      isClosed=False, color=1, thickness=line_width)
    return mask


def lanes_to_mask(
    predictions: torch.Tensor,
    img_h: int = 640,
    img_w: int = 640,
    num_points: int = 72,
    line_width: int = 8,
    conf_threshold: float = 0.4,
    max_lanes: int = 8,
) -> np.ndarray:
    """Rasterize one image's predictions into a binary mask.

    Auto-dispatches between two prediction encodings by the last dim of
    `predictions`:
      - 78-D: polyline (start_y, length, 72 x-offsets) - the P1 format.
      - 16-D: Bezier (8 control points + t_start, t_end + LCM) - the B1
        format. The 4 control point coords define the curve geometry;
        slots [13:16] hold LCM degree weights (sum to 1 when LCM on,
        zeros otherwise). When LCM is on, the curve is rendered as the
        soft mixture of K=1/2/3 Bezier curves drawn from the same 4
        control points.

    Args:
        predictions: (num_priors, 78|16) tensor of final-stage head output.
        img_h, img_w: output mask size.
        num_points: polyline-only; ignored for Bezier.
        line_width: polyline thickness.
        conf_threshold: minimum softmax(pos) score to keep.
        max_lanes: top-k cap after thresholding.

    Returns:
        np.ndarray (img_h, img_w) uint8, values in {0, 1}.
    """
    import cv2

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if predictions is None or predictions.numel() == 0:
        return mask

    # B7: dispatch to the Bezier path for 16-D predictions
    if predictions.dim() == 2 and predictions.shape[1] == 16:
        return _bezier_lanes_to_mask(
            predictions, img_h=img_h, img_w=img_w,
            line_width=line_width, conf_threshold=conf_threshold,
            max_lanes=max_lanes,
        )

    n_strips = num_points - 1
    strip_size = img_h / n_strips
    offsets_ys = np.arange(img_h, -1, -strip_size, dtype=np.float64)[:num_points]

    with torch.no_grad():
        scores = _softmax_pos_score(predictions.detach())
        kept_preds, _kept_scores = _select_topn(
            predictions, scores, conf_threshold, max_lanes,
        )
        if kept_preds.shape[0] == 0:
            return mask

    kept_np = kept_preds.detach().cpu().numpy()

    for row in kept_np:
        start = int(round(float(row[2]) * n_strips))
        start = max(0, min(start, num_points))
        # ROOT-CAUSE FIX for frozen IoU (NB88-96): the PREDICTION's length
        # field row[5] is normalized to [0, 1] - the loss trains it via
        # `reg_yxtl[:,3] = pred[5] * n_strips` against the strip-count
        # target, so pred[5] converges to ~length/71 (e.g. 0.7 for a
        # 50-strip lane). The old `int(round(row[5]))` treated it as a raw
        # strip count, so round(0.03..0.7) -> 0 or 1 -> `length < 2` ->
        # EVERY prior skipped -> near-empty mask -> IoU frozen at a per-
        # val-set floor regardless of training. `start` (row[2]) was
        # already scaled by n_strips; `length` must be too. (The GT
        # rasterizer in lane_mask_from_target.py is correct as-is because
        # GT row[5] is already in strip count.)
        length = int(round(float(row[5]) * n_strips))
        length = max(0, min(length, num_points - start))
        if length < 2:
            continue

        # CLRHead writes x values normalized to [0, 1] (divided by img_w-1).
        xs_norm = row[6 + start:6 + start + length]
        xs_pixel = xs_norm.astype(np.float64) * (img_w - 1)
        ys_pixel = offsets_ys[start:start + length]

        valid = (xs_pixel >= 0) & (xs_pixel < img_w) & (ys_pixel >= 0) & (ys_pixel < img_h)
        xs_pixel = xs_pixel[valid]
        ys_pixel = ys_pixel[valid]
        if len(xs_pixel) < 2:
            continue

        pts = np.stack([xs_pixel, ys_pixel], axis=-1).round().astype(np.int32)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)],
                      isClosed=False, color=1, thickness=line_width)
    return mask


def batch_lanes_to_mask(
    predictions: torch.Tensor,
    img_h: int = 640,
    img_w: int = 640,
    num_points: int = 72,
    line_width: int = 8,
    conf_threshold: float = 0.4,
    max_lanes: int = 8,
) -> torch.Tensor:
    """Batched wrapper: (B, num_priors, 78) -> (B, 1, H, W) float tensor."""
    masks: List[torch.Tensor] = []
    for b in range(predictions.shape[0]):
        m = lanes_to_mask(
            predictions[b], img_h=img_h, img_w=img_w, num_points=num_points,
            line_width=line_width, conf_threshold=conf_threshold,
            max_lanes=max_lanes,
        )
        masks.append(torch.from_numpy(m))
    out = torch.stack(masks, dim=0).unsqueeze(1).float()
    return out.to(predictions.device)


def smoke_test() -> int:
    print('[smoke] lane_rasterize starting', flush=True)

    num_priors = 192
    predictions = torch.full((num_priors, 78), -1e5, dtype=torch.float32)
    # Make a few "confident" lanes with valid geometry.
    for i, (sy, sx, theta, length) in enumerate(
        [(0.0, 320.0, 0.5, 40), (0.0, 100.0, 0.3, 30), (0.0, 540.0, 0.7, 30)],
    ):
        predictions[i, 0] = -5.0     # neg logit (low)
        predictions[i, 1] = 5.0      # pos logit (high) -> softmax(pos) ~ 1
        predictions[i, 2] = sy
        predictions[i, 3] = sx
        predictions[i, 4] = theta
        predictions[i, 5] = float(length)
        # x values normalized to [0, 1]:
        start_idx = int(round(sy * 71))
        x_vals = np.linspace(sx / 639.0, (sx + 60) / 639.0, length)
        predictions[i, 6 + start_idx:6 + start_idx + length] = torch.from_numpy(x_vals.astype(np.float32))
    # Make the remaining priors low-confidence.
    predictions[3:, 0] = 5.0   # neg logit high
    predictions[3:, 1] = -5.0  # pos logit low

    mask = lanes_to_mask(predictions, img_h=640, img_w=640,
                        line_width=8, conf_threshold=0.4, max_lanes=8)
    n_lane_px = int(mask.sum())
    print(f'[smoke] mask shape={mask.shape}  lane pixels={n_lane_px}  (expect > 0)')
    if mask.shape != (640, 640):
        print('[smoke] FAIL: wrong shape')
        return 1
    if n_lane_px == 0:
        print('[smoke] FAIL: empty mask, expected 3 lanes')
        return 1

    # Batched API.
    pred_batch = predictions.unsqueeze(0).repeat(2, 1, 1)
    batched = batch_lanes_to_mask(pred_batch, img_h=640, img_w=640,
                                  line_width=8, conf_threshold=0.4, max_lanes=8)
    print(f'[smoke] batched shape={tuple(batched.shape)}  (expect (2, 1, 640, 640))')
    if tuple(batched.shape) != (2, 1, 640, 640):
        print('[smoke] FAIL: batched shape')
        return 1
    if int(batched.sum()) == 0:
        print('[smoke] FAIL: batched mask empty')
        return 1

    # All-low-confidence -> empty.
    low = torch.zeros(192, 78)
    low[:, 0] = 5.0    # neg high
    low[:, 1] = -5.0   # pos low
    m_low = lanes_to_mask(low, conf_threshold=0.4)
    if int(m_low.sum()) != 0:
        print(f'[smoke] FAIL: low-confidence preds produced {int(m_low.sum())} mask pixels')
        return 1
    print('[smoke] low-confidence -> empty mask OK')

    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
