"""Phase P1 (polyline-native path): visual + IoU sanity check for the
JSON-polyline -> 78-D conversion.

For each sampled JSON, produces a 3-panel PNG:
    (left)   the source polylines (yellow) rendered on a black canvas
             at the target size
    (middle) the same canvas + slot-position dots (cyan, every 9 px in y)
             so you can see which y rows each lane occupies in the 78-D vector
    (right)  the 78-D round-trip: polylines reconstructed by reading the
             slot-aligned tensor and drawing cv2.polylines (green)

Pass criterion: dilated-IoU(left, right) >= threshold on >= 80% of samples.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bdd_label_loader import iter_records_auto  # noqa: E402
from vertices_to_clr_vector import (  # noqa: E402
    build_full_target_tensor_v2,
    select_lanes_spatial,
    vertices_to_clr_format,
)


def _draw_polylines_bgr(
    canvas: np.ndarray,
    polylines: List[np.ndarray],
    img_shape: Tuple[int, int],
    target_h: int,
    target_w: int,
    color=(0, 255, 255),  # BGR yellow
    thickness: int = 3,
) -> np.ndarray:
    """Draw each source polyline onto canvas at target size."""
    import cv2
    img_h, img_w = img_shape
    sx = target_w / img_w
    sy = target_h / img_h
    for verts in polylines:
        pts_t = np.empty_like(verts, dtype=np.float64)
        pts_t[:, 0] = verts[:, 0] * sx
        pts_t[:, 1] = verts[:, 1] * sy
        pts = pts_t.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
    return canvas


def _rasterize_target_slot_aligned(
    target: np.ndarray,
    target_h: int,
    target_w: int,
    num_points: int = 72,
    thickness: int = 3,
) -> np.ndarray:
    """Reverse of vertices_to_clr_format: reads the slot-aligned 78-D rows
    and draws each as a continuous polyline."""
    import cv2
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    n_strips = num_points - 1
    strip_size = target_h / n_strips
    offsets_ys = np.arange(target_h, -1, -strip_size, dtype=np.float64)[:num_points]

    for r in range(target.shape[0]):
        row = target[r]
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
        valid = (xs > -1e4) & (ys > -1e4)
        xs = xs[valid]
        ys = ys[valid]
        if len(xs) < 2:
            continue
        pts = np.stack([xs, ys], axis=-1).round().astype(np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)],
                      isClosed=False, color=1, thickness=thickness)
    return canvas


def verify_record(
    stem: str,
    polylines: List[np.ndarray],
    out_dir: Path,
    img_shape=(720, 1280),
    target_h: int = 640,
    target_w: int = 640,
    dilate_k: int = 7,
    max_lanes: int = 8,
) -> dict:
    """Render the 3-panel comparison for one record. Returns metrics dict.

    Color convention:
      yellow polylines = polylines selected into the (max_lanes, 78) tensor
      dim gray        = polylines DROPPED because max_lanes was full
      green           = polylines reconstructed by reading the tensor back
      cyan dots       = sample-grid slot positions (middle panel)
    """
    import cv2
    if not polylines:
        return {'stem': stem, 'ok': False, 'reason': 'no_polylines_kept'}

    # Figure out which polylines actually make it into the tensor.
    picked = select_lanes_spatial(polylines, max_lanes=max_lanes)
    picked_ids = set(id(p) for p in picked)
    dropped = [p for p in polylines if id(p) not in picked_ids]

    target = build_full_target_tensor_v2(
        polylines, img_shape=img_shape,
        target_h=target_h, target_w=target_w,
        num_points=72, max_lanes=max_lanes,
        selection='spatial',
    )

    # SOURCE rendering: dropped first (background), then picked in yellow on top.
    src_bgr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if dropped:
        _draw_polylines_bgr(src_bgr, dropped, img_shape, target_h, target_w,
                            color=(80, 80, 80), thickness=2)   # BGR dim gray
    _draw_polylines_bgr(src_bgr, picked, img_shape, target_h, target_w,
                        color=(0, 255, 255), thickness=3)       # BGR yellow

    # MIDDLE rendering: src + slot-position cyan dots showing each lane's
    # slot range, to make slot-alignment errors visible.
    mid_bgr = src_bgr.copy()
    n_strips = 71
    strip_size = target_h / n_strips
    offsets_ys = np.arange(target_h, -1, -strip_size, dtype=np.float64)[:72]
    for r in range(target.shape[0]):
        row = target[r]
        if float(row[1]) < 0.5:
            continue
        start = int(round(float(row[2]) * n_strips))
        length = int(round(float(row[5])))
        for i in range(length):
            slot = start + i
            if slot >= 72:
                break
            x = float(row[6 + slot])
            y = float(offsets_ys[slot])
            if x <= -1e4:
                continue
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < target_w and 0 <= yi < target_h:
                cv2.circle(mid_bgr, (xi, yi), 2, (255, 255, 0), -1)  # BGR cyan

    # ROUND-TRIP rendering: re-rasterize from 78-D tensor.
    reraster = _rasterize_target_slot_aligned(target, target_h, target_w)
    right_bgr = np.stack([
        np.zeros_like(reraster), reraster * 255, np.zeros_like(reraster),
    ], axis=-1)

    # Dilated IoU between source-as-mask and round-trip.
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    src_bin = (src_gray > 0).astype(np.uint8)
    kernel = np.ones((dilate_k, dilate_k), dtype=np.uint8)
    src_dil = cv2.dilate(src_bin, kernel)
    rer_dil = cv2.dilate(reraster, kernel)
    inter = int((src_dil & rer_dil).sum())
    union = int((src_dil | rer_dil).sum())
    iou = inter / union if union > 0 else 0.0
    src_covered = int((src_bin & rer_dil).sum())
    rer_covered = int((reraster & src_dil).sum())
    src_recall = src_covered / max(1, int(src_bin.sum()))
    rer_precision = rer_covered / max(1, int(reraster.sum()))

    panel = np.concatenate([src_bgr, mid_bgr, right_bgr], axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (stem + '_verify.png')
    cv2.imwrite(str(out_path), panel)

    n_lanes_encoded = int((target[:, 1] > 0.5).sum())
    return {
        'stem': stem,
        'ok': True,
        'n_polylines_loaded': len(polylines),
        'n_picked': len(picked),
        'n_dropped': len(dropped),
        'n_lanes_encoded': n_lanes_encoded,
        'iou_dilated': float(iou),
        'source_recall': float(src_recall),
        'raster_precision': float(rer_precision),
        'out_path': str(out_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='Path to BDD labels: a v1 .json file OR a v2 directory.')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_samples', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--img_h', type=int, default=720)
    p.add_argument('--img_w', type=int, default=1280)
    p.add_argument('--target_h', type=int, default=640)
    p.add_argument('--target_w', type=int, default=640)
    p.add_argument('--iou_threshold', type=float, default=0.30)
    p.add_argument('--min_pass_frac', type=float, default=0.80)
    p.add_argument('--dilate_k', type=int, default=7)
    p.add_argument('--max_lanes', type=int, default=8,
                   help='Must match the value used by convert_bdd_labels.py.')
    args = p.parse_args()

    rng = random.Random(args.seed)
    in_path = Path(args.input)
    if not in_path.exists():
        print(f'No input found at {in_path}', file=sys.stderr)
        return 2

    # Reservoir-sample N records from the iterator. Works for both v1 (one
    # big JSON stream of records) and v2 (directory of small JSONs).
    # Note: for v1 we parse the whole JSON once, so the "reservoir" really
    # just gets a random subset; for v2 we iterate file paths so it stays
    # cheap.
    reservoir: List[Tuple[str, list]] = []
    for i, (stem, polys) in enumerate(iter_records_auto(in_path)):
        if i < args.n_samples:
            reservoir.append((stem, polys))
        else:
            j = rng.randint(0, i)
            if j < args.n_samples:
                reservoir[j] = (stem, polys)

    if not reservoir:
        print(f'No records produced from {in_path}', file=sys.stderr)
        return 2

    print(f'Verifying {len(reservoir)} sampled records from {in_path}')
    good = 0
    usable = 0
    for stem, polys in reservoir:
        r = verify_record(stem, polys, Path(args.out_dir),
                          img_shape=(args.img_h, args.img_w),
                          target_h=args.target_h, target_w=args.target_w,
                          dilate_k=args.dilate_k,
                          max_lanes=args.max_lanes)
        if not r['ok']:
            print(f'  [SKIP] {stem}  reason={r["reason"]}')
            continue
        usable += 1
        if r['iou_dilated'] >= args.iou_threshold:
            good += 1
            tag = 'OK '
        else:
            tag = 'BAD'
        print(f'  [{tag}] {stem}  '
              f'loaded={r["n_polylines_loaded"]}  '
              f'picked={r["n_picked"]}  dropped={r["n_dropped"]}  '
              f'encoded={r["n_lanes_encoded"]}  '
              f'iou={r["iou_dilated"]:.3f}  '
              f'src_recall={r["source_recall"]:.3f}  '
              f'rer_prec={r["raster_precision"]:.3f}')

    print()
    pass_frac = good / max(1, usable)
    print(f'Result: {good} / {usable} usable samples pass IoU >= {args.iou_threshold} '
          f'({pass_frac*100:.1f}%; need >= {args.min_pass_frac*100:.0f}%)')
    print(f'Overlay PNGs written to: {args.out_dir}')
    return 0 if pass_frac >= args.min_pass_frac and usable > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
