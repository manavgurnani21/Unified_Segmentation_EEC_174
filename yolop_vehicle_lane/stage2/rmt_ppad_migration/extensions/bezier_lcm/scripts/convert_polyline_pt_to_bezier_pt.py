"""Phase B1 driver: convert (max_lanes, 78) polyline .pt files to
(max_lanes, 16) cubic-Bezier .pt files in parallel.

Source layout (output of `prepare_bdd_subset.py`):
    <subset_root>/lane_targets/{train2017,val2017}/<stem>.pt

Destination layout (output of this script):
    <subset_root>/lane_targets_bezier/{train2017,val2017}/<stem>.pt

Each 78-D row encodes a polyline. We decode it back to (x, y) coords at
the slot ys, then run `fit_cubic_bezier_numpy` and pack into a 16-D
Bezier vector. Empty / padding rows pass through as zeros.

Logs a histogram of fit errors over the train+val splits so the user
can sanity-check before training (target: median < 3 px, p95 < 8 px).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch


def _import_b1(migration_root: Path):
    p = migration_root / 'extensions' / 'bezier_lcm' / 'B1_data' / 'tools' / 'fit_bezier.py'
    spec = importlib.util.spec_from_file_location('b_b1_fit_bezier', p)
    if spec is None or spec.loader is None:
        raise ImportError(f'could not load B1 fit_bezier from {p}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('b_b1_fit_bezier', mod)
    spec.loader.exec_module(mod)
    return mod


def _polyline_row_to_points(row: np.ndarray, img_size: int = 640, num_points: int = 72) -> np.ndarray:
    """Decode a single 78-D polyline row back to (M, 2) (x, y) in target px.

    Row layout (per P1 vertices_to_clr_vector):
        row[0]   neg score (always 0)
        row[1]   pos score
        row[2]   start_y normalized
        row[3]   start_x (pixels, in target frame)
        row[4]   theta (normalized)
        row[5]   length (slot count)
        row[6+start:6+start+length] = x values at slot ys

    Returns:
        (M, 2) array of (x, y) pairs at the populated slots, in target
        frame pixel units. Returns empty array if the row is padding or
        too short.
    """
    if float(row[1]) < 0.5:
        return np.zeros((0, 2), dtype=np.float64)

    n_strips = num_points - 1
    strip_size = img_size / n_strips
    slot_ys = np.arange(img_size, -1, -strip_size, dtype=np.float64)[:num_points]

    start = int(round(float(row[2]) * n_strips))
    start = max(0, min(start, num_points))
    length = int(round(float(row[5])))
    length = max(0, min(length, num_points - start))
    if length < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xs = np.asarray(row[6 + start:6 + start + length], dtype=np.float64)
    ys = slot_ys[start:start + length]
    valid = (xs > -1e4) & (ys > -1e4)
    xs = xs[valid]; ys = ys[valid]
    if len(xs) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([xs, ys], axis=-1)


def _convert_one_file(
    src_pt: Path, dst_pt: Path, img_size: int, b1_mod,
) -> List[float]:
    """Convert one 78-D .pt to its 16-D Bezier counterpart. Returns
    list of fit errors for the converted lanes."""
    tensor = torch.load(src_pt, weights_only=True)
    if hasattr(tensor, 'numpy'):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.ndim != 2 or arr.shape[1] != 78:
        raise ValueError(f'{src_pt}: expected (N, 78) tensor, got {arr.shape}')

    max_lanes = arr.shape[0]
    out = np.zeros((max_lanes, b1_mod.BEZIER_VEC_LEN), dtype=np.float32)
    errors: List[float] = []

    for r in range(max_lanes):
        pts = _polyline_row_to_points(arr[r], img_size=img_size)
        if pts.shape[0] < 2:
            continue
        # vertices_to_bezier_format expects the polyline in ORIGINAL image
        # space + img_shape so it can rescale. The points we decoded are
        # already in TARGET space (img_size x img_size). So pass
        # img_shape=(img_size, img_size) so the rescaling is identity.
        vec, err = b1_mod.vertices_to_bezier_format(
            pts, img_shape=(img_size, img_size),
            target_h=img_size, target_w=img_size,
        )
        if not np.isfinite(err):
            continue
        out[r] = vec
        errors.append(err)

    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(out), dst_pt)
    return errors


def main() -> int:
    here = Path(__file__).resolve()
    migration_root = None
    for cand in here.parents:
        if (cand / 'P0_baseline').exists() and (cand / 'P8_train').exists():
            migration_root = cand
            break
    if migration_root is None:
        raise FileNotFoundError(f'could not locate rmt_ppad_migration/ from {here}')

    p = argparse.ArgumentParser()
    p.add_argument('--src-root', type=Path,
                   default=Path('/content/bdd_subset_10k/lane_targets'),
                   help='Directory containing {train2017,val2017}/<stem>.pt 78-D files')
    p.add_argument('--dst-root', type=Path,
                   default=Path('/content/bdd_subset_10k/lane_targets_bezier'),
                   help='Output directory for 16-D Bezier .pt files')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--report-json', type=Path,
                   default=Path('/content/bdd_subset_10k/lane_targets_bezier/fit_errors.json'))
    args = p.parse_args()

    b1_mod = _import_b1(migration_root)

    all_errors: List[float] = []
    counts = {'train2017': 0, 'val2017': 0}
    t0 = time.time()
    for split in ('train2017', 'val2017'):
        src_split = args.src_root / split
        dst_split = args.dst_root / split
        if not src_split.exists():
            print(f'[B1.convert] WARNING: missing {src_split}; skipping')
            continue
        files = sorted(src_split.glob('*.pt'))
        print(f'[B1.convert] {split}: {len(files)} files')
        for i, src in enumerate(files):
            dst = dst_split / src.name
            errs = _convert_one_file(src, dst, args.img_size, b1_mod)
            all_errors.extend(errs)
            counts[split] += 1
            if (i + 1) % 500 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                print(f'  [{split}] {i+1}/{len(files)}  {rate:.1f} files/s')

    if not all_errors:
        print('[B1.convert] ERROR: no lanes converted')
        return 1

    arr = np.array(all_errors, dtype=np.float64)
    report = {
        'n_lanes':       int(arr.size),
        'mean_px':       float(arr.mean()),
        'median_px':     float(np.median(arr)),
        'p95_px':        float(np.percentile(arr, 95)),
        'p99_px':        float(np.percentile(arr, 99)),
        'max_px':        float(arr.max()),
        'frac_lt_3px':   float((arr < 3.0).mean()),
        'frac_lt_8px':   float((arr < 8.0).mean()),
        'files':         counts,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n[B1.convert] fit error report:')
    for k, v in report.items():
        print(f'  {k:14s} {v}')
    if report['median_px'] > 5.0:
        print('[B1.convert] WARNING: median fit error > 5 px; consider revisiting '
              'P1 polyline extraction before training.')
    print('[B1.convert] DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
