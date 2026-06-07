"""Phase P1 step 1: convert per-pixel lane masks to polynomial coefficients.

Input: a 2-D binary mask (H, W) with lane pixels = 1, background = 0.
Output: a list of 3rd-order polynomial coefficient arrays (one per lane
        instance), each fit on x = poly(y) with y in pixel rows.

Pipeline (from appendix-path3-implementation-prompt.md sec 4.3):
    binary_mask --skeletonize--> 1-px-wide skeleton
                --label--> connected components (one per lane)
                --filter--> keep components with >= min_length pixels
                --polyfit--> 3rd-order polynomial x = a*y^3 + b*y^2 + c*y + d

The choice of x-as-function-of-y is deliberate: BDD lanes are typically near-
vertical strokes in image space, so they're well-behaved as functions of y but
NOT as functions of x (a vertical lane has x ~ const).
"""
from __future__ import annotations

from typing import List

import numpy as np


def _try_skimage_skeletonize(mask: np.ndarray) -> np.ndarray:
    """Use skimage.morphology.skeletonize. Falls back to scipy ONLY if skimage
    is not installed at all (ImportError); any other exception bubbles up so
    we don't silently produce wrong skeletons. Returns a binary (H, W) array."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion
        skel = np.zeros_like(mask, dtype=np.uint8)
        m = mask.astype(bool).copy()
        while m.any():
            eroded = binary_erosion(m)
            skel |= (m & ~eroded).astype(np.uint8)
            m = eroded
        return skel
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def _try_scipy_label(skel: np.ndarray):
    """Use scipy.ndimage.label with 8-connectivity for connected components.

    Critical: scipy's default `structure` for 2D is 4-connected (cross), which
    SPLITS a diagonal 1-px skeleton at every staircase step into many 2-pixel
    fragments. For our use case (diagonal lane skeletons) we MUST use the
    8-connected (3x3 all-ones) structure to keep each lane as one component.
    """
    from scipy.ndimage import label
    structure = np.ones((3, 3), dtype=np.uint8)
    return label(skel, structure=structure)


def extract_lane_polylines(
    mask: np.ndarray,
    min_length: int = 20,
    poly_order: int = 3,
    min_y_span: int = 15,
    max_aspect: float = 3.0,
    max_residual_px: float = 12.0,
) -> List[np.ndarray]:
    """Returns a list of polynomial coefficient arrays (length poly_order+1).

    Args:
        mask: (H, W) binary mask with lane pixels = 1.
        min_length: minimum pixel count for a connected component to count
            as a lane. Drops noise blobs.
        poly_order: polynomial degree for the x = f(y) fit. 3 is the
            appendix recommendation.
        min_y_span: drop components whose vertical extent (y_max - y_min) is
            smaller than this many pixels. Filters stop lines and crosswalk
            markings, which span lots of x but very little y - x = f(y) is
            a terrible parameterization for them.
        max_aspect: drop components whose (x_span / y_span) ratio exceeds
            this. Belt-and-suspenders with min_y_span for near-horizontal
            shapes.
        max_residual_px: drop components whose polyfit RMS residual exceeds
            this many pixels. Catches multi-valued-in-y shapes like V's
            (two lanes touching at the apex form one component that polyfit
            tries to bridge with a compromise curve that fits neither line).

    Returns:
        List of np.ndarray, each shape (poly_order + 1,). The first element
        is the highest-order coefficient, matching np.polyfit/np.polyval
        convention. Coefficients are in PIXEL units (no normalization here);
        normalization happens in polyline_to_clrnet_format.

    Note:
        Each returned ndarray has metadata attached via numpy array attrs:
        - .y_min, .y_max: pixel y-range where this component lived
          (so the caller knows where the lane is "valid")
        These are stored as the first 2 entries of the FOLLOWING object:
        the function actually returns a list of tuples
        (poly_coeffs, y_min, y_max). The appendix's signature says return
        just polylines; we return tuples because the y-range is essential
        for the next step and there's no other way to recover it.
    """
    if mask is None or mask.size == 0:
        return []
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    skeleton = _try_skimage_skeletonize(binary)
    labeled, n_components = _try_scipy_label(skeleton)

    polylines = []
    for k in range(1, n_components + 1):
        ys, xs = np.where(labeled == k)
        if ys.size < min_length:
            continue

        y_span = float(ys.max() - ys.min())
        x_span = float(xs.max() - xs.min())
        # Filter near-horizontal components (stop lines, crosswalks). x=f(y)
        # cannot represent them.
        if y_span < min_y_span:
            continue
        if y_span > 0 and (x_span / y_span) > max_aspect:
            continue

        # Sort by y ascending (top→bottom in image coords).
        order = np.argsort(ys)
        ys_s = ys[order].astype(np.float64)
        xs_s = xs[order].astype(np.float64)

        # Need at least poly_order + 1 unique y values to fit a polynomial
        # of degree poly_order without singular conditioning.
        n_unique_y = len(np.unique(ys_s))
        eff_order = min(poly_order, max(1, n_unique_y - 1))
        try:
            coeffs = np.polyfit(ys_s, xs_s, eff_order)
        except (np.linalg.LinAlgError, TypeError, ValueError):
            continue

        # Residual filter: catches V-shapes (two lanes touching at apex form
        # one component that polyfit fits with a smooth compromise curve
        # which matches neither true line).
        xs_pred = np.polyval(coeffs, ys_s)
        rms_residual = float(np.sqrt(np.mean((xs_s - xs_pred) ** 2)))
        if rms_residual > max_residual_px:
            continue

        # Pad coeffs with leading zeros so all returned arrays have the
        # same length (poly_order + 1). Makes downstream code uniform.
        if eff_order < poly_order:
            coeffs = np.concatenate([np.zeros(poly_order - eff_order), coeffs])
        polylines.append((coeffs, float(ys_s.min()), float(ys_s.max())))

    # Sort polylines by their bottom y (max y) descending — lanes closer to
    # the camera (larger y in image coords) first. Caller may use this to
    # respect max_lanes truncation in a sensible order.
    polylines.sort(key=lambda t: -t[2])
    return polylines


def smoke_test() -> int:
    """Hand-rolled mini test: draw a synthetic lane, recover its polynomial."""
    print('[smoke] mask_to_polylines starting', flush=True)
    import sys
    try:
        from skimage.morphology import skeletonize as _sk_skel  # noqa: F401
        print('[smoke] skimage.morphology.skeletonize import OK', flush=True)
    except Exception as e:
        print(f'[smoke] skimage import FAILED: {type(e).__name__}: {e}', flush=True)
        return 1
    try:
        from scipy.ndimage import label as _sp_label  # noqa: F401
        print('[smoke] scipy.ndimage.label import OK', flush=True)
    except Exception as e:
        print(f'[smoke] scipy import FAILED: {type(e).__name__}: {e}', flush=True)
        return 1
    H, W = 360, 640
    mask = np.zeros((H, W), dtype=np.uint8)
    # Synthetic lane: x = 200 + 0.5*y (near-vertical line).
    for y in range(50, 350):
        x = int(200 + 0.5 * y)
        if 0 <= x < W:
            for dx in (-1, 0, 1):
                if 0 <= x + dx < W:
                    mask[y, x + dx] = 1
    polys = extract_lane_polylines(mask, min_length=20, poly_order=3)
    print(f'Found {len(polys)} polylines (expected 1)')
    if not polys:
        return 1
    coeffs, y_min, y_max = polys[0]
    print(f'  y range: [{y_min:.1f}, {y_max:.1f}]')
    print(f'  coeffs (degree-3 desc): {coeffs}')
    # Test predicted x at y=200 should be near 200 + 0.5*200 = 300.
    pred = float(np.polyval(coeffs, 200.0))
    print(f'  x(y=200) predicted = {pred:.2f}  (truth = 300.00)')
    err = abs(pred - 300.0)
    if err > 2.0:
        print(f'  FAIL: error {err:.2f} > 2 px')
        return 1
    print('  PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
