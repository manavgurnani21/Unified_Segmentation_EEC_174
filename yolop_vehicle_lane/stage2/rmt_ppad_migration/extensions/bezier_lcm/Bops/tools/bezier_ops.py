"""Pure-function Bezier curve operations.

All functions accept either numpy or torch tensors (autodetected). The
torch path stays differentiable; the numpy path is for offline target
construction.

Layout convention for the 8-element control-point packing:
    [P0.x, P1.x, P2.x, P3.x, P0.y, P1.y, P2.y, P3.y]

This matches the appendix's lane-vector layout indices [2:10].
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_torch(x) -> bool:
    return _TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def _backend(x):
    return torch if _is_torch(x) else np


def _unpack_cps(cp8):
    """Unpack (..., 8) control points into 8 individual (..., 1) tensors.

    Slicing is used (not split) so gradients flow when the input is a torch
    tensor with requires_grad.
    """
    return (
        cp8[..., 0:1], cp8[..., 1:2], cp8[..., 2:3], cp8[..., 3:4],
        cp8[..., 4:5], cp8[..., 5:6], cp8[..., 6:7], cp8[..., 7:8],
    )


# ---------------------------------------------------------------------------
# Forward Bezier evaluation
# ---------------------------------------------------------------------------

def render_cubic(cp8: ArrayLike, ts: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Render a cubic Bezier at parameter values `ts`.

    Args:
        cp8: (..., 8) control points packed as [P0x..P3x, P0y..P3y].
        ts:  (..., S) parameter values in [0, 1]. Will broadcast against cp8.

    Returns:
        xs, ys: (..., S) coordinate samples (same units as the input cps).
    """
    P0x, P1x, P2x, P3x, P0y, P1y, P2y, P3y = _unpack_cps(cp8)
    u = 1 - ts
    xs = (u ** 3) * P0x + 3 * (u ** 2) * ts * P1x + 3 * u * (ts ** 2) * P2x + (ts ** 3) * P3x
    ys = (u ** 3) * P0y + 3 * (u ** 2) * ts * P1y + 3 * u * (ts ** 2) * P2y + (ts ** 3) * P3y
    return xs, ys


def render_linear(cp8: ArrayLike, ts: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """K=1: straight line P0 -> P3 (ignore P1, P2)."""
    P0x, _P1x, _P2x, P3x, P0y, _P1y, _P2y, P3y = _unpack_cps(cp8)
    u = 1 - ts
    xs = u * P0x + ts * P3x
    ys = u * P0y + ts * P3y
    return xs, ys


def render_quadratic(cp8: ArrayLike, ts: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """K=2: quadratic through midpoint of P1, P2.

    Per appendix LCM sec 3.1: P_mid = (P1 + P2) / 2.
    """
    P0x, P1x, P2x, P3x, P0y, P1y, P2y, P3y = _unpack_cps(cp8)
    Pmx = (P1x + P2x) * 0.5
    Pmy = (P1y + P2y) * 0.5
    u = 1 - ts
    xs = (u ** 2) * P0x + 2 * u * ts * Pmx + (ts ** 2) * P3x
    ys = (u ** 2) * P0y + 2 * u * ts * Pmy + (ts ** 2) * P3y
    return xs, ys


def render_mixture(
    cp8: ArrayLike, ts: ArrayLike, degree_weights: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """LCM soft mixture: w_1 * K1 + w_2 * K2 + w_3 * K3.

    Args:
        cp8:            (..., 8)
        ts:             (..., S)
        degree_weights: (..., 3) softmax weights over {K=1, K=2, K=3}
    Returns:
        xs, ys: (..., S)
    """
    w1 = degree_weights[..., 0:1]
    w2 = degree_weights[..., 1:2]
    w3 = degree_weights[..., 2:3]
    x1, y1 = render_linear(cp8, ts)
    x2, y2 = render_quadratic(cp8, ts)
    x3, y3 = render_cubic(cp8, ts)
    xs = w1 * x1 + w2 * x2 + w3 * x3
    ys = w1 * y1 + w2 * y2 + w3 * y3
    return xs, ys


def render_with_validity(
    cp8: ArrayLike, t_start: ArrayLike, t_end: ArrayLike,
    n_samples: int = 72, mode: str = "cubic",
    degree_weights: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Sample uniformly in [t_start, t_end] and render via the requested mode.

    Args:
        cp8:     (..., 8)
        t_start: (...,) or (..., 1)
        t_end:   (...,) or (..., 1)
        n_samples: number of t values to sample
        mode: 'cubic' | 'linear' | 'quadratic' | 'mixture'
        degree_weights: required when mode='mixture'
    Returns:
        xs, ys: (..., n_samples)
    """
    backend = _backend(cp8)
    if _is_torch(cp8):
        u = torch.linspace(0.0, 1.0, n_samples, device=cp8.device, dtype=cp8.dtype)
    else:
        u = np.linspace(0.0, 1.0, n_samples).astype(np.float64)

    if hasattr(t_start, 'shape') and t_start.shape[-1:] != (1,) and len(t_start.shape) == len(cp8.shape) - 1:
        ts_lo = t_start[..., None]
        ts_hi = t_end[..., None]
    else:
        ts_lo = t_start
        ts_hi = t_end
    ts = ts_lo + (ts_hi - ts_lo) * u

    if mode == "cubic":
        return render_cubic(cp8, ts)
    if mode == "linear":
        return render_linear(cp8, ts)
    if mode == "quadratic":
        return render_quadratic(cp8, ts)
    if mode == "mixture":
        if degree_weights is None:
            raise ValueError("mode='mixture' requires degree_weights")
        return render_mixture(cp8, ts, degree_weights)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Numpy-only: least-squares fit
# ---------------------------------------------------------------------------

# Cubic Bezier basis matrix M such that
# B(t) = [t^3, t^2, t, 1] @ M @ [P0, P1, P2, P3].T
_BEZIER_M = np.array(
    [
        [-1.0,  3.0, -3.0,  1.0],
        [ 3.0, -6.0,  3.0,  0.0],
        [-3.0,  3.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0,  0.0],
    ],
    dtype=np.float64,
)


def fit_cubic_bezier_numpy(
    points: np.ndarray, t_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Least-squares fit of a cubic Bezier to a polyline.

    Args:
        points: (N, 2) array of (x, y) coords sorted along the curve.
        t_values: optional (N,) of pre-computed t parameters. If None,
            chord-length parameterization is used.
    Returns:
        cp:       (4, 2) array of control points (P0..P3 as rows)
        fit_error: float, average L2 distance from `points` to the fit
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {points.shape}")

    n = points.shape[0]
    if n < 2:
        # degenerate: one point - return zero curve at that point
        cp = np.tile(points[0:1], (4, 1))
        return cp, 0.0
    if n < 4:
        # Few points: chord-thirds straight-line Bezier so the model has
        # SOMETHING to fit. P0 = first, P3 = last, P1/P2 at thirds.
        P0 = points[0]
        P3 = points[-1]
        P1 = (2.0 * P0 + P3) / 3.0
        P2 = (P0 + 2.0 * P3) / 3.0
        cp = np.stack([P0, P1, P2, P3], axis=0)
        ts = np.linspace(0.0, 1.0, n)
        recon_xs, recon_ys = render_cubic(_pack_4x2_to_8(cp), ts)
        fit_error = float(np.linalg.norm(
            np.stack([recon_xs, recon_ys], axis=-1) - points, axis=-1,
        ).mean())
        return cp, fit_error

    if t_values is None:
        d = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total = cum[-1]
        if total < 1e-9:
            # all points collapsed
            t_values = np.linspace(0.0, 1.0, n)
        else:
            t_values = cum / total

    T = np.stack(
        [t_values ** 3, t_values ** 2, t_values, np.ones_like(t_values)], axis=1
    )  # (N, 4)
    A = T @ _BEZIER_M  # (N, 4)
    P_x, _, _, _ = np.linalg.lstsq(A, points[:, 0], rcond=None)
    P_y, _, _, _ = np.linalg.lstsq(A, points[:, 1], rcond=None)
    cp = np.stack([P_x, P_y], axis=1)  # (4, 2)

    # Fix endpoints: a strict least-squares fit can drift the endpoints.
    # Pin P0 to the first input point and P3 to the last so the lane's
    # entry / exit visually match the rasterized mask. P1, P2 stay as the
    # solver determined.
    cp[0] = points[0]
    cp[3] = points[-1]

    recon = A @ cp
    fit_error = float(np.linalg.norm(recon - points, axis=-1).mean())
    return cp, fit_error


def _pack_4x2_to_8(cp4x2: np.ndarray) -> np.ndarray:
    """Pack a (4, 2) Bezier into the 8-element layout."""
    return np.concatenate([cp4x2[:, 0], cp4x2[:, 1]])


def pack_cps(cp4x2: ArrayLike) -> ArrayLike:
    """Pack a (..., 4, 2) Bezier into the 8-element layout (..., 8).

    Convention: [P0x, P1x, P2x, P3x, P0y, P1y, P2y, P3y].
    """
    backend = _backend(cp4x2)
    if _is_torch(cp4x2):
        return torch.cat([cp4x2[..., 0], cp4x2[..., 1]], dim=-1)
    return np.concatenate([cp4x2[..., 0], cp4x2[..., 1]], axis=-1)


def unpack_cps(cp8: ArrayLike) -> ArrayLike:
    """Inverse of pack_cps. (..., 8) -> (..., 4, 2)."""
    if _is_torch(cp8):
        x = cp8[..., 0:4]
        y = cp8[..., 4:8]
        return torch.stack([x, y], dim=-1)
    return np.stack([cp8[..., 0:4], cp8[..., 4:8]], axis=-1)


# ---------------------------------------------------------------------------
# Adaptive rasterization (numpy/CPU - eval-time only)
# ---------------------------------------------------------------------------

def render_adaptive_numpy(
    cp4x2: np.ndarray, t_start: float = 0.0, t_end: float = 1.0,
    max_error_pixel: float = 1.0, max_points: int = 200,
    img_size: int = 640,
) -> np.ndarray:
    """Adaptive subdivision rendering: recurse where polyline error > eps.

    Returns:
        points: (M, 2) array of (x, y) in PIXEL units
        Image-bound clipping is NOT done here; caller is responsible.
    """
    P = cp4x2.astype(np.float64)

    def eval_at(t):
        u = 1 - t
        x = u ** 3 * P[0, 0] + 3 * u ** 2 * t * P[1, 0] + 3 * u * t ** 2 * P[2, 0] + t ** 3 * P[3, 0]
        y = u ** 3 * P[0, 1] + 3 * u ** 2 * t * P[1, 1] + 3 * u * t ** 2 * P[2, 1] + t ** 3 * P[3, 1]
        return np.array([x, y], dtype=np.float64) * img_size

    points = {t_start: eval_at(t_start), t_end: eval_at(t_end)}
    stack = [(t_start, t_end)]
    while stack:
        if len(points) >= max_points:
            break
        a, b = stack.pop()
        m = 0.5 * (a + b)
        pa = points[a]
        pb = points[b]
        pm_true = eval_at(m)
        pm_lin = 0.5 * (pa + pb)
        if np.linalg.norm(pm_true - pm_lin) > max_error_pixel and (b - a) > 1e-3:
            stack.append((a, m))
            stack.append((m, b))
            points[m] = pm_true
    ts = sorted(points.keys())
    return np.stack([points[t] for t in ts], axis=0)


__all__ = [
    "render_cubic", "render_linear", "render_quadratic", "render_mixture",
    "render_with_validity",
    "fit_cubic_bezier_numpy", "pack_cps", "unpack_cps",
    "render_adaptive_numpy",
]
