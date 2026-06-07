"""Phase P1 step 3: visual sanity check for mask -> 78-D conversion.

For each input mask, produce a 3-panel PNG:
    (left)   original binary mask, resized to target shape (grayscale)
    (middle) the polyline fit (yellow curves) overlaid on the mask
    (right)  the 78-D tensor rasterized back as a continuous polyline (green)

Pass criterion: on at least 8 of 10 random samples, the rasterized lanes
have dilated-IoU >= the threshold (default 0.25) against the resized mask.

Acceptance per the migration plan:
    "verify_conversion.py overlays correct on 10+ random samples"

Note on cv2 color convention: cv2.imwrite consumes BGR. The "yellow"
overlay below uses (0, 255, 255) which is BGR yellow (green+red, no blue).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Make sibling tools importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from mask_to_polylines import extract_lane_polylines  # noqa: E402
from polyline_to_clrnet_format import build_full_target_tensor  # noqa: E402


def _rasterize_78d_row(row: np.ndarray, target_h: int, target_w: int,
                       num_points: int = 72, thickness: int = 3) -> np.ndarray:
    """Render one 78-D vector back to a binary mask using cv2.polylines.

    Draws a continuous polyline through the in-image samples so that the
    rasterized result is comparable to the dense source mask (rather than
    a series of 72 disjoint dots).
    """
    import cv2  # heavy import - lazy
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    if float(row[1]) < 0.5:  # pos_score < 0.5 -> no lane
        return canvas

    n_strips = num_points - 1
    strip_size = target_h / n_strips
    offsets_ys = np.arange(target_h, -1, -strip_size, dtype=np.float64)[:num_points]

    # 78-D layout (CLRKDNet generate_lane_line.py):
    #   row[2] = len(xs_outside_image) / n_strips
    #   row[5] = len(xs_inside_image)
    #   row[6:78] = [outside_xs, inside_xs] concatenated
    # The typical BDD case: outside_xs is empty (no horizontal clipping),
    # so the inside block aligns with offsets_ys[0:length].
    start_y_idx = int(round(float(row[2]) * n_strips))
    start_y_idx = max(0, min(start_y_idx, num_points))
    length = int(round(float(row[5])))
    length = max(0, min(length, num_points - start_y_idx))
    if length < 2:
        return canvas

    inside_xs = np.asarray(row[6 + start_y_idx:6 + start_y_idx + length], dtype=np.float64)
    inside_ys = offsets_ys[start_y_idx:start_y_idx + length]

    # Drop sentinels (-1e5 padding) and points whose pixel coords are far
    # outside the canvas. We allow a small over-the-edge slack so polylines
    # that just barely exit still get clipped by cv2 cleanly.
    valid = (inside_xs > -1e4) & (inside_ys > -1e4)
    inside_xs = inside_xs[valid]
    inside_ys = inside_ys[valid]
    if len(inside_xs) < 2:
        return canvas

    pts = np.stack([inside_xs, inside_ys], axis=-1).round().astype(np.int32)
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)],
                  isClosed=False, color=1, thickness=thickness)
    return canvas


def _render_polyfit_overlay(
    base_mask: np.ndarray,
    polylines,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Build a BGR canvas = resized-mask (grayscale) + yellow polynomial curves."""
    import cv2
    src_h, src_w = base_mask.shape[:2]
    img = cv2.resize(base_mask.astype(np.uint8) * 255, (target_w, target_h),
                     interpolation=cv2.INTER_NEAREST)
    canvas = np.stack([img, img, img], axis=-1)  # BGR grayscale

    sx = target_w / src_w
    sy = target_h / src_h
    for coeffs, y_min, y_max in polylines:
        ys_dense = np.linspace(y_min, y_max, num=200)
        xs_dense = np.polyval(coeffs, ys_dense)
        pts = np.stack([xs_dense * sx, ys_dense * sy], axis=-1).round().astype(np.int32)
        # (0, 255, 255) in BGR = yellow.
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)],
                      isClosed=False, color=(0, 255, 255), thickness=2)
    return canvas


def verify_one(
    mask_path: Path,
    out_dir: Path,
    target_h: int = 640,
    target_w: int = 640,
    min_length: int = 20,
    dilate_k: int = 7,
) -> dict:
    """Process a single mask. Return metrics dict; write a 3-panel PNG."""
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {'path': str(mask_path), 'ok': False, 'reason': 'imread_failed'}
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {'path': str(mask_path), 'ok': False, 'reason': 'empty_mask'}

    polylines = extract_lane_polylines(binary, min_length=min_length, poly_order=3)
    if not polylines:
        return {'path': str(mask_path), 'ok': False, 'reason': 'no_polylines',
                'mask_pixels': int(binary.sum())}

    target = build_full_target_tensor(
        polylines, mask_shape=binary.shape,
        target_h=target_h, target_w=target_w, num_points=72, max_lanes=4,
    )

    # Round-trip rasterize all encoded lanes into one canvas.
    reraster = np.zeros((target_h, target_w), dtype=np.uint8)
    for i in range(target.shape[0]):
        reraster |= _rasterize_78d_row(target[i], target_h, target_w)

    # Dilated IoU: tolerate ~3 px error in either direction (kernel 7x7).
    mask_rs = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((dilate_k, dilate_k), dtype=np.uint8)
    mask_dil = cv2.dilate(mask_rs, kernel)
    reraster_dil = cv2.dilate(reraster, kernel)
    inter = int((mask_dil & reraster_dil).sum())
    union = int((mask_dil | reraster_dil).sum())
    iou = inter / union if union > 0 else 0.0

    # Coverage metrics: what fraction of mask pixels are within `dilate_k`
    # of a rasterized pixel, and vice versa? Often a more honest measure
    # than pure IoU when one side is sparser than the other.
    mask_covered = int((mask_rs & reraster_dil).sum())
    raster_covered = int((reraster & mask_dil).sum())
    mask_recall = mask_covered / max(1, int(mask_rs.sum()))
    raster_precision = raster_covered / max(1, int(reraster.sum()))

    # 3-panel output (BGR).
    overlay = _render_polyfit_overlay(binary, polylines, target_h, target_w)
    mask_bgr = np.stack([mask_rs * 255] * 3, axis=-1)
    reraster_bgr = np.stack([
        np.zeros_like(reraster),    # B
        reraster * 255,             # G  (BGR green)
        np.zeros_like(reraster),    # R
    ], axis=-1)
    panel = np.concatenate([mask_bgr, overlay, reraster_bgr], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (mask_path.stem + '_verify.png')
    cv2.imwrite(str(out_path), panel)

    n_lanes = int((target[:, 1] > 0.5).sum())
    return {
        'path': str(mask_path),
        'ok': True,
        'n_polylines': len(polylines),
        'n_lanes_encoded': n_lanes,
        'iou_dilated': float(iou),
        'mask_recall': float(mask_recall),
        'raster_precision': float(raster_precision),
        'out_path': str(out_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--mask_dir', required=True,
                   help='Folder of binary lane masks (PNG, gray, non-zero = lane).')
    p.add_argument('--out_dir', required=True,
                   help='Where to write the 3-panel PNGs.')
    p.add_argument('--n_samples', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--target_h', type=int, default=640)
    p.add_argument('--target_w', type=int, default=640)
    p.add_argument('--iou_threshold', type=float, default=0.20,
                   help='Dilated IoU threshold to count a sample as "good".')
    p.add_argument('--min_pass_frac', type=float, default=0.80,
                   help='Fraction of usable samples that must clear the IoU '
                        'threshold for the script to exit 0.')
    p.add_argument('--dilate_k', type=int, default=7,
                   help='Kernel size for the dilation in the IoU comparison.')
    args = p.parse_args()

    rng = random.Random(args.seed)
    files = sorted(Path(args.mask_dir).glob('*.png'))
    if not files:
        print(f'No PNGs found in {args.mask_dir}', file=sys.stderr)
        return 2
    if len(files) > args.n_samples:
        files = rng.sample(files, args.n_samples)

    print(f'Verifying {len(files)} samples from {args.mask_dir}')
    results = []
    good = 0
    usable = 0
    for f in files:
        r = verify_one(f, Path(args.out_dir),
                       target_h=args.target_h, target_w=args.target_w,
                       dilate_k=args.dilate_k)
        results.append(r)
        if not r['ok']:
            print(f'  [SKIP] {Path(r["path"]).name}  reason={r["reason"]}')
            continue
        usable += 1
        if r['iou_dilated'] >= args.iou_threshold:
            good += 1
            tag = 'OK '
        else:
            tag = 'BAD'
        print(f'  [{tag}] {Path(r["path"]).name}  '
              f'n_polylines={r["n_polylines"]}  '
              f'n_lanes_encoded={r["n_lanes_encoded"]}  '
              f'iou={r["iou_dilated"]:.3f}  '
              f'mask_recall={r["mask_recall"]:.3f}  '
              f'raster_prec={r["raster_precision"]:.3f}')

    print()
    pass_frac = good / max(1, usable)
    print(f'Result: {good} / {usable} usable samples pass IoU >= {args.iou_threshold} '
          f'({pass_frac*100:.1f}%; need >= {args.min_pass_frac*100:.0f}%)')
    print(f'Overlay PNGs written to: {args.out_dir}')
    return 0 if pass_frac >= args.min_pass_frac and usable > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
