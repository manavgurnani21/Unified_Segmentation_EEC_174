"""Phase P1 (polyline-native path): convert a polyline of (x, y) vertices
directly to a CLRHead 78-D target vector.

Replaces the mask -> skeletonize -> connected-components -> polyfit pipeline
in [polyline_to_clrnet_format.py]. The advantage:

    - Each BDD JSON polyline is already a clean ordered (x, y) sequence,
      so V-shapes and multiple touching lanes stay separate (one JSON
      object = one polyline).
    - No 4-vs-8-connectivity gotchas. No connected-components merging
      lanes that share a pixel at an intersection.
    - No global cubic-polynomial smoothing that distorts genuine curves.
      We use scipy InterpolatedUnivariateSpline which passes through every
      input vertex exactly (and degenerates to linear for <4 points).

The output 78-D format MATCHES the slot-aligned packing of the mask path
(see polyline_to_clrnet_format.py docstring): slot `out[6+i]` is the x
value at `prior_ys[i]`, with -1e5 sentinels outside the polyline's vertical
span. start_y = bottom-most slot index of the polyline. length = number of
consecutive slots covering the polyline.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def vertices_to_clr_format(
    vertices: np.ndarray,
    img_shape: Tuple[int, int],
    target_h: int = 640,
    target_w: int = 640,
    num_points: int = 72,
) -> Optional[np.ndarray]:
    """Convert one BDD polyline to a 78-D CLRHead target vector.

    Args:
        vertices: (N, 2) array of (x, y) pixel coords in ORIGINAL image
            coordinate system (typically the BDD 720x1280 frame).
        img_shape: (img_h, img_w) of the source image - used for the
            rescale to target_h x target_w.
        target_h, target_w: training image size.
        num_points: CLRHead grid resolution (72 for CULane config).

    Returns:
        np.ndarray of shape (78,) or None if the polyline is too short / too
        thin / lands fully outside the image after rescale.
    """
    if vertices is None or len(vertices) < 2:
        return None

    n_strips = num_points - 1
    strip_size = target_h / n_strips
    offsets_ys = np.arange(target_h, -1, -strip_size, dtype=np.float64)[:num_points]

    img_h, img_w = img_shape
    sx = target_w / img_w
    sy = target_h / img_h

    pts = np.asarray(vertices, dtype=np.float64)
    pts_t = np.empty_like(pts)
    pts_t[:, 0] = pts[:, 0] * sx
    pts_t[:, 1] = pts[:, 1] * sy

    # Sort by y descending so we always have a consistent "bottom-first"
    # parameterization. The spline below needs y strictly increasing, so
    # we'll reverse for that call.
    pts_t = pts_t[np.argsort(-pts_t[:, 1])]

    # Drop duplicate y values (the spline can't handle them).
    seen = set()
    keep = []
    for i, y in enumerate(pts_t[:, 1]):
        yk = float(y)
        if yk in seen:
            continue
        seen.add(yk)
        keep.append(i)
    pts_t = pts_t[keep]
    if len(pts_t) < 2:
        return None

    y_t_min = float(pts_t[:, 1].min())
    y_t_max = float(pts_t[:, 1].max())
    if y_t_max - y_t_min < strip_size:
        # Polyline is shorter than one strip - cannot occupy 2 grid slots.
        return None

    # Snap to nearest: extend the slot range by half a strip on each side
    # so we capture the offsets_y that's closest to (but maybe just outside)
    # the polyline's y endpoints. Without this snap we lose up to ~9 px of
    # vertical extent at each end (visible in NB80 panels as green-shorter-
    # than-yellow). The spline can extrapolate up to half a strip safely.
    y_margin = strip_size * 0.5
    inside_grid = (offsets_ys >= y_t_min - y_margin) & (offsets_ys <= y_t_max + y_margin)
    inside_idx = np.where(inside_grid)[0]
    if len(inside_idx) < 2:
        return None
    start = int(inside_idx.min())
    end = int(inside_idx.max())
    length = end - start + 1

    slot_ys_t = offsets_ys[start:start + length]
    # Clip the y values we feed to the spline to the polyline's true domain
    # so extrapolation is bounded - if a slot's y is 4 px above y_t_min, we
    # evaluate the spline at y_t_min (the polyline's actual end) and use
    # that x for the slot. Cleaner than a true cubic extrapolation that
    # can shoot off-screen.
    slot_ys_for_spline = np.clip(slot_ys_t, y_t_min, y_t_max)

    from scipy.interpolate import InterpolatedUnivariateSpline
    ys_inc = pts_t[::-1, 1]
    xs_inc = pts_t[::-1, 0]
    k = min(3, len(pts_t) - 1)
    try:
        spline = InterpolatedUnivariateSpline(ys_inc, xs_inc, k=k)
        slot_xs_t = np.asarray(spline(slot_ys_for_spline), dtype=np.float32)
    except Exception:
        return None

    in_image = (slot_xs_t >= 0) & (slot_xs_t < target_w)
    if int(in_image.sum()) < 2:
        return None

    vec_len = 2 + 4 + num_points
    out = np.full(vec_len, -1e5, dtype=np.float32)
    out[0] = 0.0
    out[1] = 1.0
    out[2] = start / n_strips
    out[3] = float(slot_xs_t[0])

    thetas: List[float] = []
    for i in range(1, length):
        denom = (float(slot_xs_t[i]) - float(slot_xs_t[0])) + 1e-5
        t = math.atan(i * strip_size / denom) / math.pi
        if t <= 0:
            t = 1 - abs(t)
        thetas.append(t)
    out[4] = float(np.mean(thetas)) if thetas else 0.5
    out[5] = float(length)

    out[6 + start:6 + start + length] = slot_xs_t
    return out


def select_lanes_spatial(
    polylines: Sequence[np.ndarray],
    max_lanes: int = 8,
) -> List[np.ndarray]:
    """Pick up to max_lanes polylines maximizing road-cross spatial coverage.

    Algorithm:
      1. Start by picking the polyline with the largest arc length (longest
         lane = strongest signal for the model).
      2. Repeatedly pick the polyline whose MEAN X is the furthest from any
         already-picked polyline's mean x. Maximizes left/right coverage.
      3. Stop when max_lanes are picked or no candidates remain.

    Why arc length first (not y_span)? BDD often encodes a single physical
    lane as several `poly2d` dashes. Sorting purely by y_span makes the
    top-N picks cluster around ego (where lanes are visually tall). Sorting
    by arc length and then enforcing x-diversity gives ego-left, ego-right,
    far-left, far-right - the four most informative samples of a road.
    """
    if not polylines:
        return []
    stats = []
    for verts in polylines:
        arc = float(np.sqrt(((np.diff(verts, axis=0)) ** 2).sum(axis=1)).sum())
        mean_x = float(verts[:, 0].mean())
        stats.append((arc, mean_x, verts))
    stats.sort(key=lambda s: -s[0])

    picked = [stats[0]]
    remaining = stats[1:]
    while remaining and len(picked) < max_lanes:
        picked_xs = [s[1] for s in picked]

        def _dist_to_picked(s):
            return min(abs(s[1] - px) for px in picked_xs)

        best = max(remaining, key=_dist_to_picked)
        picked.append(best)
        remaining = [s for s in remaining if s is not best]
    return [s[2] for s in picked]


def build_full_target_tensor_v2(
    polylines: Sequence[np.ndarray],
    img_shape: Tuple[int, int],
    target_h: int = 640,
    target_w: int = 640,
    num_points: int = 72,
    max_lanes: int = 8,
    selection: str = 'spatial',
) -> np.ndarray:
    """Pack up to max_lanes polylines into a (max_lanes, 78) target tensor.

    polylines: list of (N, 2) ndarrays in ORIGINAL image pixel coords.
    selection:
        'spatial' (default): use select_lanes_spatial - arc-length first
            pick + spatial-diversity for subsequent picks. Best for BDD.
        'first':  legacy first-N behavior (whatever order the caller passed).
            Useful if you've pre-sorted upstream.
    """
    if selection == 'spatial':
        chosen = select_lanes_spatial(polylines, max_lanes=max_lanes)
    elif selection == 'first':
        chosen = list(polylines)[:max_lanes]
    else:
        raise ValueError(f"selection must be 'spatial' or 'first', got {selection!r}")

    vec_len = 2 + 4 + num_points
    out = np.full((max_lanes, vec_len), -1e5, dtype=np.float32)
    out[:, 0] = 1.0
    out[:, 1] = 0.0

    written = 0
    for verts in chosen:
        if written >= max_lanes:
            break
        vec = vertices_to_clr_format(
            verts, img_shape=img_shape,
            target_h=target_h, target_w=target_w, num_points=num_points,
        )
        if vec is None:
            continue
        out[written] = vec
        written += 1
    return out


def smoke_test() -> int:
    """Hand-rolled test: straight diagonal polyline through a 720x1280 frame."""
    import sys
    print('[smoke] vertices_to_clr_vector starting', flush=True)
    try:
        from scipy.interpolate import InterpolatedUnivariateSpline  # noqa: F401
        print('[smoke] scipy.interpolate import OK', flush=True)
    except Exception as e:
        print(f'[smoke] scipy import FAILED: {type(e).__name__}: {e}', flush=True)
        return 1

    # Diagonal lane: x = 400 + 0.6*y, y in [100, 700]
    verts = np.array([
        [400 + 0.6 * y, y] for y in (100, 250, 400, 550, 700)
    ], dtype=np.float64)
    vec = vertices_to_clr_format(verts, img_shape=(720, 1280),
                                 target_h=640, target_w=640, num_points=72)
    if vec is None:
        print('[smoke] FAIL: vertices_to_clr_format returned None')
        return 1
    print(f'[smoke] vec.shape={vec.shape}  pos={vec[1]:.1f}  '
          f'start_y={vec[2]:.3f}  start_x={vec[3]:.1f}  length={vec[5]:.1f}')
    if vec.shape != (78,):
        print('[smoke] FAIL: wrong vec shape')
        return 1
    if not (vec[0] == 0 and vec[1] == 1):
        print('[smoke] FAIL: classification scores wrong')
        return 1
    if vec[5] < 30:
        print(f'[smoke] FAIL: length {vec[5]} too small (lane should span most of image)')
        return 1

    # Slot-alignment sanity: the x-values in the populated slot range should
    # all be inside the target image width.
    start = int(round(vec[2] * 71))
    length = int(round(vec[5]))
    xs = vec[6 + start:6 + start + length]
    in_range = ((xs >= 0) & (xs < 640)).sum()
    print(f'[smoke] populated slots [{start}:{start + length}); '
          f'{int(in_range)}/{length} x in [0, 640)')
    if in_range < length * 0.9:
        print('[smoke] FAIL: too many x values outside image')
        return 1

    # Build full target (4, 78) and check no-lane rows are sentinel.
    full = build_full_target_tensor_v2([verts], img_shape=(720, 1280))
    print(f'[smoke] full target shape={full.shape}')
    if full[1, 0] != 1 or full[1, 1] != 0:
        print('[smoke] FAIL: no-lane row scores wrong')
        return 1
    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
