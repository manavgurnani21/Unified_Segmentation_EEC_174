"""Phase P6: rasterize a (max_lanes, 78) lane-target tensor back to a
binary (H, W) mask. Used as the `lane_seg_mask` channel for P5's
`lane_seg_aux_loss` (cross-entropy on CLRHead's internal aux seg head).

Same logic the P3 + P1 verifier panels use; centralized here so the
dataset loader (P6) and the validator (P7) share one implementation.

Slot-aligned packing per P1:
    row[2] = start_y       (slot index of the lane's bottom-most y / n_strips)
    row[5] = length        (number of CONSECUTIVE slots the lane occupies)
    row[6+start:6+start+length] = x values (pixel coords, in target image)
"""
from __future__ import annotations

from typing import Union

import numpy as np


def rasterize_lane_target_to_mask(
    target,
    img_h: int = 640,
    img_w: int = 640,
    num_points: int = 72,
    thickness: int = 4,
) -> np.ndarray:
    """Build a binary (H, W) mask by drawing each encoded lane.

    Auto-dispatches between two encodings by the last dim of `target`:
      - 78-D: P1 polyline target (start_y, length, 72 x-offsets)
      - 16-D: B1 Bezier target (8 control points + 2 validity + ...)

    target: torch.Tensor or np.ndarray of shape (max_lanes, 78|16).
            Rows where row[1] < 0.5 are skipped.

    Returns: np.ndarray, shape (img_h, img_w), dtype uint8, values in {0, 1}.
    """
    import cv2

    # Accept torch or numpy.
    if hasattr(target, 'detach'):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.asarray(target)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if target_np.size == 0:
        return mask

    # B1: detect 16-D Bezier targets and rasterize via cubic Bezier
    if target_np.ndim == 2 and target_np.shape[1] == 16:
        return _rasterize_bezier(target_np, img_h, img_w, thickness)

    n_strips = num_points - 1
    strip_size = img_h / n_strips
    offsets_ys = np.arange(img_h, -1, -strip_size, dtype=np.float64)[:num_points]

    for r in range(target_np.shape[0]):
        row = target_np[r]
        if float(row[1]) < 0.5:
            continue
        start = int(round(float(row[2]) * n_strips))
        start = max(0, min(start, num_points))
        length = int(round(float(row[5])))
        length = max(0, min(length, num_points - start))
        if length < 2:
            continue

        xs = np.asarray(row[6 + start:6 + start + length], dtype=np.float64)
        ys = offsets_ys[start:start + length]
        # `np.isfinite` is the key guard: the old `> -1e4` mask filtered the
        # -1e5 sentinels but let +Inf / NaN through, and `.astype(np.int32)`
        # on a non-finite value raises "RuntimeWarning: invalid value
        # encountered in cast" and produces a garbage pixel coord. Keep only
        # finite, in-a-sane-range points before the int cast.
        valid = (
            np.isfinite(xs) & np.isfinite(ys)
            & (xs > -1e4) & (ys > -1e4) & (xs < 1e4) & (ys < 1e4)
        )
        xs = xs[valid]
        ys = ys[valid]
        if len(xs) < 2:
            continue

        pts = np.stack([xs, ys], axis=-1).round().astype(np.int32)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], isClosed=False,
                      color=1, thickness=thickness)
    return mask


def _rasterize_bezier(
    target_np: np.ndarray,
    img_h: int, img_w: int, thickness: int,
    n_render: int = 100,
) -> np.ndarray:
    """Bezier path: draw each row's cubic curve as a polyline."""
    import cv2

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    ts = np.linspace(0.0, 1.0, n_render)

    for r in range(target_np.shape[0]):
        row = target_np[r]
        if float(row[1]) < 0.5:
            continue
        # Layout per fit_bezier.py:
        #   row[2:6]  = P0..P3 x  (normalized)
        #   row[6:10] = P0..P3 y  (normalized)
        #   row[10]   = t_start
        #   row[11]   = t_end
        cps_x = row[2:6].astype(np.float64)
        cps_y = row[6:10].astype(np.float64)
        t_start = float(row[10])
        t_end = float(row[11])
        if t_end - t_start < 0.01:
            continue
        t = t_start + (t_end - t_start) * ts
        u = 1 - t
        xs = (u**3 * cps_x[0] + 3*u**2*t * cps_x[1]
              + 3*u*t**2 * cps_x[2] + t**3 * cps_x[3]) * (img_w - 1)
        ys = (u**3 * cps_y[0] + 3*u**2*t * cps_y[1]
              + 3*u*t**2 * cps_y[2] + t**3 * cps_y[3]) * (img_h - 1)
        pts = np.stack([xs, ys], axis=-1).round().astype(np.int32)
        # Filter points outside the frame so cv2.polylines doesn't draw
        # off-image strokes that wrap.
        inside = (
            (pts[:, 0] >= 0) & (pts[:, 0] < img_w)
            & (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
        )
        if inside.sum() < 2:
            continue
        pts = pts[inside]
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], isClosed=False,
                      color=1, thickness=thickness)
    return mask


def smoke_test() -> int:
    print('[smoke] lane_mask_from_target starting', flush=True)
    # Build a synthetic target with two diagonal lanes.
    target = np.full((8, 78), -1e5, dtype=np.float32)
    target[:, 0] = 1.0
    target[:, 1] = 0.0
    # Lane 0: bottom-up vertical-ish at x=200.
    target[0, 0] = 0.0
    target[0, 1] = 1.0
    target[0, 2] = 0.0   # start_y -> bottom of image
    target[0, 5] = 30.0  # 30 slots upward
    target[0, 6:6 + 30] = np.linspace(200, 250, 30)
    # Lane 1: mid-image diagonal at x=400-500.
    target[1, 0] = 0.0
    target[1, 1] = 1.0
    target[1, 2] = 0.3   # start_y -> slot ~21 (mid)
    start1 = int(round(0.3 * 71))
    target[1, 5] = 20.0
    target[1, 6 + start1:6 + start1 + 20] = np.linspace(400, 500, 20)

    mask = rasterize_lane_target_to_mask(target, img_h=640, img_w=640, thickness=4)
    print(f'[smoke] mask shape {mask.shape}, dtype {mask.dtype}, '
          f'pixels-set {int(mask.sum())}  (expect > 0)')
    if mask.shape != (640, 640):
        print('[smoke] FAIL: wrong shape')
        return 1
    if int(mask.sum()) == 0:
        print('[smoke] FAIL: empty mask')
        return 1

    # No-lane row check.
    empty = np.full((8, 78), -1e5, dtype=np.float32)
    empty[:, 0] = 1.0
    empty[:, 1] = 0.0
    m2 = rasterize_lane_target_to_mask(empty, img_h=640, img_w=640)
    if int(m2.sum()) != 0:
        print('[smoke] FAIL: no-lane target produced non-empty mask')
        return 1
    print('[smoke] no-lane target -> empty mask OK')

    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
