"""Phase P1 (polyline-native path): batch-convert BDD lane labels to .pt
target tensors.

Auto-detects the input format per --split_inputs arg:
  - file path ending .json  -> BDD v1 monolithic per-split JSON
  - directory               -> BDD v2 per-image JSONs

Writes one <out_root>/<out_split_name>/<image_stem>.pt per record in the
input, even if the record had zero kept polylines (then the .pt is all
no-lane sentinels). This makes downstream image<->target pairing 1:1 by
filename stem.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bdd_label_loader import iter_records_auto  # noqa: E402
from vertices_to_clr_vector import build_full_target_tensor_v2  # noqa: E402


def convert_split(
    input_path: Path,
    out_dir: Path,
    img_shape=(720, 1280),
    target_h: int = 640,
    target_w: int = 640,
    num_points: int = 72,
    max_lanes: int = 4,
    progress_every: int = 5000,
) -> dict:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Converting BDD labels from {input_path} -> {out_dir}')
    t0 = time.time()

    n_written = 0
    n_empty = 0
    n_failed = 0
    n_lanes_hist = np.zeros(max_lanes + 1, dtype=np.int64)

    no_lane = np.full((max_lanes, 2 + 4 + num_points), -1e5, dtype=np.float32)
    no_lane[:, 0] = 1.0
    no_lane[:, 1] = 0.0

    for i, (stem, polys) in enumerate(iter_records_auto(input_path)):
        if not stem:
            n_failed += 1
            continue

        if not polys:
            target = no_lane.copy()
            n_empty += 1
        else:
            target = build_full_target_tensor_v2(
                polys, img_shape=img_shape,
                target_h=target_h, target_w=target_w,
                num_points=num_points, max_lanes=max_lanes,
            )
            if int((target[:, 1] > 0.5).sum()) == 0:
                n_empty += 1

        n_lanes = int((target[:, 1] > 0.5).sum())
        n_lanes_hist[n_lanes] += 1

        out_path = out_dir / (stem + '.pt')
        torch.save(torch.from_numpy(target), out_path)
        n_written += 1

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f'  [{i + 1}] elapsed={elapsed:.0f}s  '
                  f'empty={n_empty}  failed={n_failed}  '
                  f'rate={(i + 1) / max(elapsed, 1e-6):.0f}/s')

    elapsed = time.time() - t0
    print(f'Done: {n_written} written, {n_empty} empty, {n_failed} failed, '
          f'in {elapsed:.0f}s ({n_written / max(elapsed, 1e-6):.1f}/s)')
    print(f'Lanes-per-image histogram: {n_lanes_hist.tolist()}  '
          f'(indices 0..{max_lanes})')
    return {
        'total': n_written,
        'written': n_written,
        'empty': n_empty,
        'failed': n_failed,
        'lanes_hist': n_lanes_hist.tolist(),
        'elapsed_s': elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--train_input', required=True,
                   help='Path to train labels: a v1 .json file OR a v2 directory.')
    p.add_argument('--val_input', required=True,
                   help='Path to val labels: a v1 .json file OR a v2 directory.')
    p.add_argument('--out_root', required=True,
                   help='Output folder. Per-split subfolders will be created.')
    p.add_argument('--train_out_name', default='train2017')
    p.add_argument('--val_out_name', default='val2017')
    p.add_argument('--img_h', type=int, default=720)
    p.add_argument('--img_w', type=int, default=1280)
    p.add_argument('--target_h', type=int, default=640)
    p.add_argument('--target_w', type=int, default=640)
    p.add_argument('--num_points', type=int, default=72)
    p.add_argument('--max_lanes', type=int, default=8,
                   help='Per-image lane capacity in the (max_lanes, 78) target '
                        'tensor. CULane uses 4 but BDD scenes routinely have '
                        '5-8 markings, so we bump to 8. Pure data-shape change; '
                        'CLRHead`s 192 anchor priors and dynamic matching are '
                        'unaffected. P7 NMS top-k must match this value.')
    args = p.parse_args()

    out_root = Path(args.out_root)

    summary = {}
    for (in_path_str, out_name) in (
        (args.train_input, args.train_out_name),
        (args.val_input,   args.val_out_name),
    ):
        in_path = Path(in_path_str)
        out_dir = out_root / out_name
        if not in_path.exists():
            print(f'SKIP: {in_path} does not exist')
            continue
        s = convert_split(
            in_path, out_dir,
            img_shape=(args.img_h, args.img_w),
            target_h=args.target_h, target_w=args.target_w,
            num_points=args.num_points, max_lanes=args.max_lanes,
        )
        summary[out_name] = s

    summary_path = out_root / 'conversion_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\nSummary -> {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
