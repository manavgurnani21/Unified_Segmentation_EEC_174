"""Phase P1 step 4: batch-convert BDD lane masks to .pt files of 78-D targets.

For every PNG under <mask_root>/<split>/*.png (e.g.
/content/BDD_seg_mask/mask/lane/train2017/*.png), write
<out_root>/<split>/<stem>.pt containing a torch.Tensor of shape (4, 78).

The .pt files are written to LOCAL Colab storage (default <out_root>=
/content/lane_targets) per the Drive-vs-local rule in the migration README:
tens of thousands of small files MUST NOT go to /content/drive/...

After this script finishes, the user should tarball <out_root> and copy
that tarball back to Drive for persistence:
    tar -C /content -czf /content/drive/.../lane_targets.tar.gz lane_targets
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_to_polylines import extract_lane_polylines  # noqa: E402
from polyline_to_clrnet_format import build_full_target_tensor  # noqa: E402


def convert_split(
    mask_dir: Path,
    out_dir: Path,
    target_h: int = 640,
    target_w: int = 640,
    num_points: int = 72,
    max_lanes: int = 4,
    min_length: int = 20,
    progress_every: int = 500,
) -> dict:
    """Convert every PNG in mask_dir, writing one .pt per file to out_dir.

    Skips files where the encoder produced 0 lanes (writes the file anyway
    so the dataloader can find it, but logs them).
    """
    import cv2
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(mask_dir.glob('*.png'))
    n_total = len(files)
    if n_total == 0:
        print(f'WARN: no PNGs in {mask_dir}', file=sys.stderr)
        return {'total': 0, 'written': 0, 'empty': 0, 'failed': 0}

    print(f'Converting {n_total} masks from {mask_dir} -> {out_dir}')
    t0 = time.time()

    n_written = 0
    n_empty = 0
    n_failed = 0
    n_lanes_hist = np.zeros(max_lanes + 1, dtype=np.int64)

    # Pre-build the no-lane fallback target so we can drop in a placeholder
    # whenever a mask is missing/corrupt/empty. Keeping a 1:1 file-count
    # between .png and .pt is required by NB80's acceptance check.
    no_lane = np.full((max_lanes, 2 + 4 + num_points), -1e5, dtype=np.float32)
    no_lane[:, 0] = 1.0
    no_lane[:, 1] = 0.0

    for i, f in enumerate(files):
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            n_failed += 1
            target = no_lane.copy()
        else:
            binary = (mask > 0).astype(np.uint8)
            if binary.sum() == 0:
                target = no_lane.copy()
                n_empty += 1
            else:
                polys = extract_lane_polylines(binary, min_length=min_length, poly_order=3)
                target = build_full_target_tensor(
                    polys, mask_shape=binary.shape,
                    target_h=target_h, target_w=target_w,
                    num_points=num_points, max_lanes=max_lanes,
                )
                if int((target[:, 1] > 0.5).sum()) == 0:
                    # mask had pixels but skeletonize/polyfit produced nothing
                    n_empty += 1

        n_lanes = int((target[:, 1] > 0.5).sum())
        n_lanes_hist[n_lanes] += 1

        out_path = out_dir / (f.stem + '.pt')
        torch.save(torch.from_numpy(target), out_path)
        n_written += 1

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - (i + 1))
            print(f'  [{i + 1}/{n_total}] elapsed={elapsed:.0f}s  eta={eta:.0f}s  '
                  f'empty={n_empty}  failed={n_failed}')

    elapsed = time.time() - t0
    print(f'Done: {n_written} written, {n_empty} empty, {n_failed} failed, '
          f'in {elapsed:.0f}s ({n_written / max(elapsed, 1e-6):.1f}/s)')
    print(f'Lanes-per-image histogram: {n_lanes_hist.tolist()}  '
          f'(indices 0..{max_lanes} = num lanes encoded)')
    return {
        'total': n_total,
        'written': n_written,
        'empty': n_empty,
        'failed': n_failed,
        'lanes_hist': n_lanes_hist.tolist(),
        'elapsed_s': elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--mask_root', required=True,
                   help='Folder containing per-split subfolders of masks, e.g. '
                        '/content/BDD_seg_mask/mask/lane')
    p.add_argument('--out_root', required=True,
                   help='Output folder. Per-split subfolders will be created.')
    p.add_argument('--splits', default='train2017,val2017')
    p.add_argument('--target_h', type=int, default=640)
    p.add_argument('--target_w', type=int, default=640)
    p.add_argument('--num_points', type=int, default=72)
    p.add_argument('--max_lanes', type=int, default=4)
    p.add_argument('--min_length', type=int, default=20)
    args = p.parse_args()

    mask_root = Path(args.mask_root)
    out_root = Path(args.out_root)

    summary = {}
    for split in args.splits.split(','):
        split = split.strip()
        if not split:
            continue
        mask_dir = mask_root / split
        out_dir = out_root / split
        if not mask_dir.exists():
            print(f'SKIP: {mask_dir} does not exist')
            continue
        s = convert_split(
            mask_dir, out_dir,
            target_h=args.target_h, target_w=args.target_w,
            num_points=args.num_points, max_lanes=args.max_lanes,
            min_length=args.min_length,
        )
        summary[split] = s

    import json
    summary_path = out_root / 'conversion_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\nSummary -> {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
