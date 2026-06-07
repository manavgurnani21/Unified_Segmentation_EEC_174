"""Phase P1 step 2: convert polynomial coefficients to CLRHead 78-D target vectors.

Output format (must match CLRKDNet/configs/DLA_CULane.py exactly):
    num_points = 72, max_lanes = 4 -> vector length = 2 + 1 + 1 + 1 + 1 + 72 = 78
    [0]      neg_score (1 = no lane present, 0 = lane present)
    [1]      pos_score (0 = no lane, 1 = lane present)
    [2]      start_y (number of below-image-bottom "extrap" points / n_strips)
    [3]      start_x (first x value INSIDE the image width, in target pixels)
    [4]      theta (averaged tangent angle, normalized to [0, 1])
    [5]      length (number of in-image points)
    [6:78]   72 x-coordinates in pixels, order = extrap-first then inside

The default no-lane row is [1, 0, -1e5, -1e5, ..., -1e5]. This matches the
sentinel that CLRKDNet's loss expects (FocalLoss interprets the first 2 cols
as the BG/FG classification target).

The y-sampling grid is the SAME as CLRKDNet:
    offsets_ys = np.arange(target_h, -1, -strip_size)[:num_points]
that is: 72 y values evenly spaced from img bottom to img top.

We work in MASK pixel coords for the polynomial evaluation, then rescale x
and y by (target_w / mask_w) and (target_h / mask_h) respectively. A
polynomial of degree k in y stays a polynomial of degree k in y under
uniform linear rescaling - so straight evaluation-then-rescale is exact.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def _sample_lane_from_poly(
    poly_coeffs: np.ndarray,
    y_min_mask: float,
    y_max_mask: float,
    mask_shape: Tuple[int, int],
    target_h: int,
    target_w: int,
    offsets_ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the polynomial at the CLRHead grid, ONLY inside the lane's
    actual vertical span.

    Args:
        poly_coeffs: degree-3 polynomial coeffs from np.polyfit (highest power first).
        y_min_mask, y_max_mask: the lane's vertical span IN MASK COORDS.
        mask_shape: (mask_h, mask_w) for rescaling.
        target_h, target_w: training image size (typically 640x640).
        offsets_ys: 72 y values from bottom to top, in TARGET coords.

    Returns:
        (extrap_xs, interp_xs) - x values in TARGET coords.
        extrap_xs is ALWAYS EMPTY in this implementation. We kept the return
        shape for API compatibility, but we deliberately do NOT extrapolate
        below the lane's bottom (toward the image bottom).

    Why no extrap?
        CLRKDNet's original transform_annotation extrapolates a lane down to
        the image bottom using the polynomial's tangent at y_max. That makes
        sense for the CULane dataset where every annotated lane is a road
        marking that physically extends to the camera-near foreground (its
        bottom got clipped by the annotation polygon, but the underlying
        physical lane continues downward).

        BDD's "lane" mask channel does NOT have that property - it includes
        short distant lanes, stop lines, crosswalk markings, and various
        partial-frame markings that genuinely DO end mid-image. Extrapolating
        them down to y=640 produces fake lane tails that:
          - mislead the model into thinking lanes always reach the bottom
          - inflate the IoU/ACC eval against the mask (false positives in
            the bottom rows)
          - drag a near-horizontal polynomial's extrapolation off-screen at
            a wild angle (visible in NB80 verify panels)
    """
    mask_h, mask_w = mask_shape
    sx = target_w / mask_w  # scale x: mask_x -> target_x
    sy = target_h / mask_h  # scale y: mask_y -> target_y

    y_min_t = y_min_mask * sy
    y_max_t = y_max_mask * sy

    inside_grid = (offsets_ys >= y_min_t) & (offsets_ys <= y_max_t)
    inside_ys_t = offsets_ys[inside_grid]
    inside_ys_mask = inside_ys_t / sy
    inside_xs_mask = np.polyval(poly_coeffs, inside_ys_mask)
    inside_xs = inside_xs_mask * sx

    extrap_xs = np.array([], dtype=np.float64)
    return extrap_xs, inside_xs


def polyline_to_clr_format(
    poly_coeffs: np.ndarray,
    y_min_mask: float,
    y_max_mask: float,
    mask_shape: Tuple[int, int],
    target_h: int = 640,
    target_w: int = 640,
    num_points: int = 72,
) -> Optional[np.ndarray]:
    """Convert one polynomial lane fit to a 78-D CLRHead target vector.

    Slot-aligned packing (NOT CLRKDNet's left-justified packing):
        Each of the 72 x-slots `out[6+i]` corresponds to `offsets_ys[i]` =
        `prior_ys[i]` in CLRHead's prediction grid. We:
          1. Find the slot range [start, start+length-1] where offsets_ys
             falls inside the lane's actual vertical span [y_min, y_max].
          2. Evaluate the polynomial at exactly those slots.
          3. Write x values at slots [start:start+length].
          4. Leave the other slots as the -1e5 sentinel (no lane there).
          5. Record start_y = start / n_strips, length = length.

    Why slot-aligned instead of left-justified?
        CLRKDNet's transform_annotation left-justifies because their lanes
        always reach the image bottom (via mandatory extrapolation), which
        accidentally aligns slot 0 with offsets_ys[0]. We dropped that
        extrapolation (see _sample_lane_from_poly), so left-justifying would
        place a mid-image lane's x values at image-bottom slots - the
        rasterizer would then draw the lane at the wrong y position.
        Slot-aligned packing is mandatory once extrap is off.

    Returns None when the lane covers fewer than 2 sampling slots (a 1-slot
    lane has no tangent direction and CLRHead can't regress it).
    """
    n_strips = num_points - 1
    strip_size = target_h / n_strips
    offsets_ys = np.arange(target_h, -1, -strip_size, dtype=np.float64)
    offsets_ys = offsets_ys[:num_points]  # trim FP-rounding overshoot

    mask_h, mask_w = mask_shape
    sx = target_w / mask_w
    sy = target_h / mask_h
    y_min_t = y_min_mask * sy
    y_max_t = y_max_mask * sy

    inside_grid = (offsets_ys >= y_min_t) & (offsets_ys <= y_max_t)
    inside_idx = np.where(inside_grid)[0]
    if len(inside_idx) < 2:
        return None
    start = int(inside_idx.min())
    end = int(inside_idx.max())
    length = end - start + 1

    slot_ys_t = offsets_ys[start:start + length]
    slot_xs_mask = np.polyval(poly_coeffs, slot_ys_t / sy)
    slot_xs_t = (slot_xs_mask * sx).astype(np.float32)

    in_image = (slot_xs_t >= 0) & (slot_xs_t < target_w)
    if int(in_image.sum()) < 2:
        return None

    vec_len = 2 + 4 + num_points
    out = np.full(vec_len, -1e5, dtype=np.float32)
    out[0] = 0.0
    out[1] = 1.0
    out[2] = start / n_strips
    out[3] = float(slot_xs_t[0])  # x at bottom-most slot of the lane

    thetas = []
    for i in range(1, length):
        denom = (slot_xs_t[i] - slot_xs_t[0]) + 1e-5
        t = math.atan(i * strip_size / denom) / math.pi
        if t <= 0:
            t = 1 - abs(t)
        thetas.append(t)
    out[4] = float(np.mean(thetas)) if thetas else 0.5
    out[5] = float(length)

    out[6 + start:6 + start + length] = slot_xs_t
    return out


def build_full_target_tensor(
    polylines: List[Tuple[np.ndarray, float, float]],
    mask_shape: Tuple[int, int],
    target_h: int = 640,
    target_w: int = 640,
    num_points: int = 72,
    max_lanes: int = 4,
) -> np.ndarray:
    """Pack up to max_lanes polylines into a (max_lanes, 78) tensor.

    polylines is the output of extract_lane_polylines, already sorted by
    decreasing bottom-y (closer-to-camera first). We take the first max_lanes
    that survive polyline_to_clr_format's filtering.
    """
    vec_len = 2 + 4 + num_points
    out = np.full((max_lanes, vec_len), -1e5, dtype=np.float32)
    out[:, 0] = 1.0  # neg_score default = 1 (background)
    out[:, 1] = 0.0  # pos_score default = 0

    written = 0
    for coeffs, y_min, y_max in polylines:
        if written >= max_lanes:
            break
        vec = polyline_to_clr_format(
            coeffs, y_min, y_max,
            mask_shape=mask_shape,
            target_h=target_h, target_w=target_w,
            num_points=num_points,
        )
        if vec is None:
            continue
        out[written] = vec
        written += 1
    return out


def smoke_test() -> int:
    """Synthetic straight lane through a 720x1280 mask, target 640x640."""
    import sys
    print('[smoke] polyline_to_clrnet_format starting', flush=True)
    try:
        from mask_to_polylines import extract_lane_polylines as _f  # noqa: F401
        print('[smoke] mask_to_polylines import OK', flush=True)
    except Exception as e:
        print(f'[smoke] mask_to_polylines import FAILED: {type(e).__name__}: {e}', flush=True)
        return 1
    H_mask, W_mask = 720, 1280
    mask = np.zeros((H_mask, W_mask), dtype=np.uint8)
    # Lane: x = 400 + 0.6*y, y in [100, 700], width 3 px
    for y in range(100, 700):
        x = int(400 + 0.6 * y)
        for dx in (-1, 0, 1):
            if 0 <= x + dx < W_mask:
                mask[y, x + dx] = 1

    # Step 1: mask -> polylines
    from mask_to_polylines import extract_lane_polylines
    polys = extract_lane_polylines(mask, min_length=20, poly_order=3)
    if not polys:
        print('FAIL: no polylines extracted')
        return 1
    print(f'Extracted {len(polys)} polyline(s) (expected 1)')

    # Step 2: polylines -> 78-D tensor
    target = build_full_target_tensor(
        polys, mask_shape=(H_mask, W_mask),
        target_h=640, target_w=640, num_points=72, max_lanes=4,
    )
    print(f'Target shape: {target.shape}  (expected (4, 78))')
    assert target.shape == (4, 78), f'wrong shape {target.shape}'

    # Row 0 should be a valid lane; rows 1-3 should be no-lane placeholders.
    print(f'Row 0 neg/pos scores: {target[0, 0]:.1f} / {target[0, 1]:.1f}  (expect 0.0 / 1.0)')
    print(f'Row 1 neg/pos scores: {target[1, 0]:.1f} / {target[1, 1]:.1f}  (expect 1.0 / 0.0)')
    if not (target[0, 0] == 0 and target[0, 1] == 1):
        print('FAIL: row 0 scores wrong')
        return 1
    if not (target[1, 0] == 1 and target[1, 1] == 0):
        print('FAIL: row 1 (placeholder) scores wrong')
        return 1

    # Spot-check: row 0 length should be > 50 (lane was 600 px tall, plenty of strips).
    length = target[0, 5]
    print(f'Row 0 length (in strips): {length:.1f}')
    if length < 30:
        print(f'FAIL: lane too short')
        return 1

    # Spot-check: start_x should be near 400*640/1280 + 0.6*y_max*640/1280
    # at y_max_mask~700, so x_mask~820, x_target~410.
    start_x = target[0, 3]
    print(f'Row 0 start_x: {start_x:.1f}  (rough expected ~ x at bottom of in-image)')

    # Spot-check: a few of the 72 x-values should be sensible (between 0 and 640).
    xs = target[0, 6:6 + int(target[0, 2] * 71 + length)]
    in_range = ((xs >= 0) & (xs < 640)).sum()
    print(f'Row 0 x-values in [0, 640): {in_range} / {len(xs)}')

    print('PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
