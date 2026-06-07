"""CLRKDNet-style per-lane curve F1 - the metric the references actually use.

WHY: our headline metric is a merged-mask pixel IoU, which saturates ~0.09
for thin polylines vs thin GT lines even when the curves are good - so a
plateauing pixel-IoU does NOT mean the model stopped improving. Curve F1
measures "did we find each lane?": decode each predicted lane and each GT
lane to its OWN thick polyline mask, match pred<->GT by per-lane IoU, and
count TP / FP / FN at an IoU threshold. This is monotone with real lane
quality and is the right yardstick for comparing polyline vs bezier heads.

Encoding (verified): the PRED length (row[5]) is normalized -> *n_strips;
PRED x-values are normalized -> *(R-1). The GT length is strip-count; GT
x-values are pixels in the 640 frame -> *(R/640). Both handled below.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _decode_lane_polyline(row, R: int, num_points: int = 72,
                          is_pred: bool = True) -> Optional[np.ndarray]:
    """One 78-D lane row -> (M,2) int polyline at resolution R, or None.

    GT row[1] is a 0/1 validity flag (gate on it). PRED row[1] is a raw cls
    LOGIT that is routinely < 0.5 even for a good-geometry prior, so gating
    preds on it drops almost every predicted lane -> curveIoU pinned at 0.
    The proven merged rasterizer (lane_rasterize.lanes_to_mask) never does
    this; it ranks by softmax score and renders geometry regardless. Mirror
    that: only gate GT here (the caller already ranks/selects preds).
    """
    if not is_pred and float(row[1]) < 0.5:
        return None
    n_strips = num_points - 1
    ys = np.arange(R, -1, -R / n_strips, dtype=np.float64)[:num_points]
    start = int(round(float(row[2]) * n_strips))
    start = max(0, min(start, num_points))
    raw_len = float(row[5])
    length = int(round(raw_len * n_strips)) if is_pred else int(round(raw_len))
    length = max(0, min(length, num_points - start))
    if length < 2:
        return None
    xs = np.asarray(row[6 + start:6 + start + length], dtype=np.float64)
    xs_px = xs * (R - 1) if is_pred else xs * (R / 640.0)
    yy = ys[start:start + length]
    pts = [(int(round(x)), int(round(y))) for x, y in zip(xs_px, yy)
           if np.isfinite(x) and -1e4 < x and 0 <= x < R and 0 <= y < R]
    if len(pts) < 2:
        return None
    return np.array(pts, dtype=np.int32)


def _decode_lane_bezier(row, R: int, num_points: int = 72,
                        is_pred: bool = True) -> Optional[np.ndarray]:
    """One 16-D Bezier lane row -> (M,2) int polyline at resolution R, or None.

    Both PRED and GT Bezier control points live in normalized [0,1] x [0,1]
    (the fit and the head both work in that frame), so there is no
    pred/GT scaling split like the polyline path needs. Renders the pure
    cubic over the predicted validity range [t_start, t_end]; the LCM
    degree-mixture is a rendering refinement that does not change which
    lane was found, so the metric uses the cubic.

    SPRINT-0 FIX (NEXT_STEPS_lane_head_strategy S0.1): only GT row[1] is a
    0/1 validity flag. PRED row[1] is a raw cls LOGIT, routinely < 0.5 even
    for a good-geometry curve, so gating preds on it pinned bezier curve-F1
    at 0 for all 30 epochs (the bezier runs were mis-scored / abandoned).
    Mirror the polyline decoder: only gate GT; the caller ranks preds by
    softmax score and renders geometry regardless.
    """
    if not is_pred and float(row[1]) < 0.5:
        return None
    cps_x = np.asarray(row[2:6], dtype=np.float64)
    cps_y = np.asarray(row[6:10], dtype=np.float64)
    t_start = float(row[10])
    t_end = float(row[11])
    if not (np.isfinite(t_start) and np.isfinite(t_end)) or (t_end - t_start) < 0.02:
        return None
    t = np.linspace(t_start, t_end, num_points)
    u = 1.0 - t
    xs = (u**3 * cps_x[0] + 3*u**2*t * cps_x[1]
          + 3*u*t**2 * cps_x[2] + t**3 * cps_x[3]) * (R - 1)
    ys = (u**3 * cps_y[0] + 3*u**2*t * cps_y[1]
          + 3*u*t**2 * cps_y[2] + t**3 * cps_y[3]) * (R - 1)
    pts = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)
           if np.isfinite(x) and np.isfinite(y) and 0 <= x < R and 0 <= y < R]
    if len(pts) < 2:
        return None
    return np.array(pts, dtype=np.int32)


def _decode_row(row, R: int, num_points: int, is_pred: bool) -> Optional[np.ndarray]:
    """Dispatch by vector width: 78-D polyline vs 16-D Bezier."""
    if len(row) == 16:
        return _decode_lane_bezier(row, R, num_points, is_pred=is_pred)
    return _decode_lane_polyline(row, R, num_points, is_pred=is_pred)


def _rasterize_one(poly: np.ndarray, R: int, width: int):
    import cv2
    m = np.zeros((R, R), dtype=np.uint8)
    cv2.polylines(m, [poly.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=width)
    return m.astype(bool)


def _softmax_pos(rows: np.ndarray) -> np.ndarray:
    z = rows[:, :2]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def lane_curve_tp_fp_fn(pred_rows, gt_rows, R: int = 160, num_points: int = 72,
                        line_width: int = 12, iou_thr: float = 0.5,
                        max_lanes: int = 8, curveiou_pool: int = 64,
                        conf_tau: float = 0.0) -> tuple:
    """Return (tp, fp, fn, best_iou_sum, n_gt) for one image.

    Two DIFFERENT pred sets on purpose:
      - F1 (tp/fp/fn @ iou_thr) uses the top-`max_lanes` by pos score -- the
        real detection metric, which depends on a working cls head.
      - curveIoU (best_iou_sum / n_gt) uses a BROAD pool (top-`curveiou_pool`)
        so it measures lane GEOMETRY independent of the cls ranking. Early on
        cls is near-flat (score_std ~0.07) so the top-8 are a fixed clustered
        set that can miss every GT -> a top-8-gated curveIoU pins at 0 even as
        geometry improves. The broad pool is the intended un-saturated signal.

    pred_rows: (P, 78|16) np array (final-stage predictions, eval frame).
    gt_rows:   (max_lanes, 78|16) np array (P1 / B1 targets).
    """
    pred_rows = np.asarray(pred_rows, dtype=np.float64)
    gt_rows = np.asarray(gt_rows, dtype=np.float64)

    # Rank once by pos score (descending). F1 uses the strict top-max_lanes;
    # curveIoU uses a BROAD pool (top-curveiou_pool) so it tracks geometry
    # independent of the (often near-flat early) cls ranking -- otherwise the
    # same clustered top-8 are scored every epoch and curveIoU pins at 0.
    if pred_rows.shape[0] > 1:
        order = np.argsort(-_softmax_pos(pred_rows))
        pred_rows = pred_rows[order]
    # S1.1 confidence threshold: keep only preds with softmax_pos > conf_tau for
    # the F1 set (the curveIoU pool stays broad/un-thresholded so geometry is
    # still tracked). tau=0 reproduces the old top-max_lanes behavior exactly.
    # This makes the metric honor a decode threshold == NB103/validator decode,
    # so the "always 8 lanes" failure is fixable without retraining.
    if conf_tau > 0.0 and pred_rows.shape[0] > 0:
        scores_sorted = _softmax_pos(pred_rows)
        f1_rows = pred_rows[scores_sorted > conf_tau]
    else:
        f1_rows = pred_rows

    gt_masks = []
    for row in gt_rows:
        g = _decode_row(row, R, num_points, is_pred=False)
        if g is not None:
            gt_masks.append(_rasterize_one(g, R, line_width))
    n_gt = len(gt_masks)

    def _rasterize(rows_subset):
        out = []
        for row in rows_subset:
            p = _decode_row(row, R, num_points, is_pred=True)
            if p is not None:
                out.append(_rasterize_one(p, R, line_width))
        return out

    def _iou_mat(gms, pms):
        mat = np.zeros((len(gms), len(pms)), dtype=np.float64)
        for i, gm in enumerate(gms):
            ga = gm.sum()
            for j, pm in enumerate(pms):
                inter = np.logical_and(gm, pm).sum()
                union = ga + pm.sum() - inter
                mat[i, j] = inter / union if union > 0 else 0.0
        return mat

    # F1 pred set: strict top-max_lanes by score (after the S1.1 tau filter).
    f1_masks = _rasterize(f1_rows[:max_lanes])
    n_pred = len(f1_masks)
    if n_gt == 0:
        return 0, n_pred, 0, 0.0, 0      # all preds are false positives

    # curveIoU: best-match IoU per GT over a BROAD pool (geometry tracker).
    pool_masks = _rasterize(pred_rows[:curveiou_pool])
    best_iou_sum = float(_iou_mat(gt_masks, pool_masks).max(axis=1).sum()) \
        if pool_masks else 0.0

    if n_pred == 0:
        return 0, 0, n_gt, best_iou_sum, n_gt   # all GTs missed by the top-N

    # F1: greedy 1-to-1 match on the top-N set, TP/FP/FN @ iou_thr.
    iou = _iou_mat(gt_masks, f1_masks)
    tp = 0
    used_p, used_g = set(), set()
    flat = sorted(((iou[i, j], i, j) for i in range(n_gt) for j in range(n_pred)),
                  reverse=True)
    for v, i, j in flat:
        if v < iou_thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j); tp += 1
    fp = n_pred - len(used_p)
    fn = n_gt - len(used_g)
    return tp, fp, fn, best_iou_sum, n_gt


def f1_from_counts(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'f1': f1, 'precision': prec, 'recall': rec, 'tp': tp, 'fp': fp, 'fn': fn}


__all__ = ['lane_curve_tp_fp_fn', 'f1_from_counts']
