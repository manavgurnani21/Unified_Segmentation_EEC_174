"""Phase B1: convert BDD polylines into 16-D cubic-Bezier target tensors.

Pipeline (mirrors P1's polyline path but with a different per-lane vector):

  1. Reuse `select_lanes_spatial` from P1's vertices_to_clr_vector to pick
     up to max_lanes polylines per image (longest arc-length first +
     spatial diversity).
  2. For each chosen polyline:
        - rescale BDD pixel coords (img_h, img_w) -> (target_h, target_w)
        - least-squares fit a cubic Bezier
        - normalize control points to [0, 1] x [0, 1] (with a [-0.2, 1.2]
          tolerance window for lanes that overshoot the frame edge, per
          appendix sec 4.2)
        - compute a heuristic complexity score (P1, P2 deviation from chord)
  3. Pack into a (max_lanes, 16) float32 tensor matching the appendix
     layout:

        [ 0]      neg score  (always 0 here; trainer renormalizes if needed)
        [ 1]      pos score  (1 if real lane, else 0)
        [ 2..5]   P0.x, P1.x, P2.x, P3.x   normalized in [-0.2, 1.2]
        [ 6..9]   P0.y, P1.y, P2.y, P3.y   normalized in [-0.2, 1.2]
        [10]      t_start   (always 0.0 in v1)
        [11]      t_end     (always 1.0 in v1)
        [12]      heuristic curve_complexity in [0, 1]
        [13..15]  reserved (LCM stores degree weights here at training time;
                  during data conversion they stay 0.0)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Import siblings (P1's helpers + Bops's fitter) by absolute path so this
# script works whether it's run from the repo root, from `extensions/`, or
# loaded from /content/ in Colab.
# ---------------------------------------------------------------------------

def _find_migration_root() -> Path:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            return cur
        cur = cur.parent
    # Fallback: walk up to repo root then descend
    for ancestor in here.parents:
        cand = ancestor / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration'
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not locate rmt_ppad_migration/ from {here}"
    )


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


_MIGRATION = _find_migration_root()
_p1_v2c = _import_by_path(
    'b1_p1_vertices_helpers',
    _MIGRATION / 'P1_data_conversion' / 'tools' / 'vertices_to_clr_vector.py',
)
_bops = _import_by_path(
    'b1_bezier_ops',
    _MIGRATION / 'extensions' / 'bezier_lcm' / 'Bops' / 'tools' / 'bezier_ops.py',
)

select_lanes_spatial = _p1_v2c.select_lanes_spatial  # type: ignore[attr-defined]
fit_cubic_bezier_numpy = _bops.fit_cubic_bezier_numpy  # type: ignore[attr-defined]
render_cubic = _bops.render_cubic  # type: ignore[attr-defined]
pack_cps = _bops.pack_cps  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Per-lane Bezier vector
# ---------------------------------------------------------------------------

BEZIER_VEC_LEN = 16
CONTROL_RANGE_MIN = -0.2
CONTROL_RANGE_MAX = 1.2


def vertices_to_bezier_format(
    vertices: np.ndarray,
    img_shape: Tuple[int, int],
    target_h: int = 640,
    target_w: int = 640,
) -> Tuple[np.ndarray, float]:
    """Convert ONE polyline into a 16-D Bezier vector.

    Args:
        vertices: (N, 2) array of (x, y) BDD pixel coords.
        img_shape: (img_h, img_w) of the source frame.
    Returns:
        vec: (16,) float32 vector matching the layout in this module's
            docstring.
        fit_error: average L2 px distance between fitted Bezier and input
            polyline, in TARGET-frame pixels. Useful for filtering noisy
            lanes upstream (caller may discard if > threshold).
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        return np.zeros(BEZIER_VEC_LEN, dtype=np.float32), float('inf')

    img_h, img_w = img_shape
    sx = target_w / float(img_w)
    sy = target_h / float(img_h)
    pts_t = pts * np.array([sx, sy])  # (N, 2) in target frame

    cp4x2, fit_err = fit_cubic_bezier_numpy(pts_t)

    # Normalize to [0, 1] then clip with a tolerance window.
    cp_normalized = cp4x2 / np.array([target_w - 1.0, target_h - 1.0])
    cp_normalized = np.clip(cp_normalized, CONTROL_RANGE_MIN, CONTROL_RANGE_MAX)

    # Heuristic curve-complexity score: distance of P1, P2 from the chord
    # midpoint of P0-P3, in normalized coords. Used as a soft target if
    # the model is trained to predict slot [12].
    P0 = cp_normalized[0]
    P3 = cp_normalized[3]
    chord_mid = 0.5 * (P0 + P3)
    P1_dev = float(np.linalg.norm(cp_normalized[1] - chord_mid))
    P2_dev = float(np.linalg.norm(cp_normalized[2] - chord_mid))
    complexity = min((P1_dev + P2_dev), 1.0)

    vec = np.zeros(BEZIER_VEC_LEN, dtype=np.float32)
    vec[0] = 0.0  # neg score
    vec[1] = 1.0  # pos score
    vec[2:6] = cp_normalized[:, 0]   # P0.x..P3.x
    vec[6:10] = cp_normalized[:, 1]  # P0.y..P3.y
    vec[10] = 0.0  # t_start
    vec[11] = 1.0  # t_end
    vec[12] = float(complexity)
    # vec[13..15] reserved
    return vec, fit_err


def build_bezier_target_tensor(
    polylines: Sequence[np.ndarray],
    img_shape: Tuple[int, int],
    target_h: int = 640,
    target_w: int = 640,
    max_lanes: int = 8,
    fit_error_threshold_px: float = 25.0,
    selection: str = 'spatial',
) -> Tuple[np.ndarray, List[float]]:
    """Pack up to max_lanes Bezier fits into a (max_lanes, 16) tensor.

    Args:
        polylines: list of (N, 2) ndarrays in ORIGINAL image px coords.
        fit_error_threshold_px: drop fits with average error above this
            (in target-frame pixels). A high default (25 px on a 640
            frame) lets nearly all real BDD lanes through; it primarily
            filters degenerate polylines that the spline fit hallucinates.
        selection: 'spatial' (= P1 default) or 'first'.
    Returns:
        out:      (max_lanes, 16) float32 tensor. Unused slots are zero
                  (their pos-score=0 marks them as padding).
        fit_errors: list of fit errors for the lanes actually written.
    """
    if selection == 'spatial':
        chosen = select_lanes_spatial(polylines, max_lanes=max_lanes)
    elif selection == 'first':
        chosen = list(polylines)[:max_lanes]
    else:
        raise ValueError(f"selection must be 'spatial' or 'first'; got {selection!r}")

    out = np.zeros((max_lanes, BEZIER_VEC_LEN), dtype=np.float32)
    errors: List[float] = []
    written = 0
    for verts in chosen:
        if written >= max_lanes:
            break
        vec, err = vertices_to_bezier_format(
            verts, img_shape=img_shape,
            target_h=target_h, target_w=target_w,
        )
        if not np.isfinite(err) or err > fit_error_threshold_px:
            continue
        out[written] = vec
        errors.append(err)
        written += 1
    return out, errors


# ---------------------------------------------------------------------------
# CLI smoke test: synthetic curves
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    """Sanity-check the fit on a synthetic cubic curve."""
    print('[B1.smoke] generating synthetic curved polyline')
    ts = np.linspace(0, 1, 50)
    # Known cubic: P0=(100, 600), P1=(150, 200), P2=(500, 200), P3=(550, 600)
    P0 = np.array([100.0, 600.0]); P1 = np.array([150.0, 200.0])
    P2 = np.array([500.0, 200.0]); P3 = np.array([550.0, 600.0])
    u = 1 - ts
    xs = u**3 * P0[0] + 3 * u**2 * ts * P1[0] + 3 * u * ts**2 * P2[0] + ts**3 * P3[0]
    ys = u**3 * P0[1] + 3 * u**2 * ts * P1[1] + 3 * u * ts**2 * P2[1] + ts**3 * P3[1]
    poly = np.stack([xs, ys], axis=1)

    vec, err = vertices_to_bezier_format(
        poly, img_shape=(720, 1280), target_h=640, target_w=640,
    )
    print(f'[B1.smoke] vec[1] (pos)        = {vec[1]:.2f}')
    print(f'[B1.smoke] vec[2:6] (P0..3 x)  = {vec[2:6]}')
    print(f'[B1.smoke] vec[6:10] (P0..3 y) = {vec[6:10]}')
    print(f'[B1.smoke] vec[12] (complexity)= {vec[12]:.3f}')
    print(f'[B1.smoke] fit error           = {err:.2f} px')
    assert vec[1] == 1.0
    assert err < 5.0, f'fit error {err} too high for a clean cubic input'
    assert vec[12] > 0.05, 'complexity should be non-trivial for a curved input'
    print('[B1.smoke] OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(_smoke_test())


__all__ = [
    'BEZIER_VEC_LEN', 'vertices_to_bezier_format', 'build_bezier_target_tensor',
]
