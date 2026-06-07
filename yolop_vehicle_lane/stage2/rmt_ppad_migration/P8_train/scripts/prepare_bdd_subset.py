"""Phase P8 smoke-test data prep: build a small BDD slice on /content/.

Strategy (mirrors NB79 cell 4 / P0's proven setup):
  1. Extract the project's already-prepared `bdd100k_clrkd_curve.tar` to
     /content/. It contains a clean `images/{train,val}/<stem>.jpg` layout
     produced by `stage2/scripts/04_prepare_bdd_curve_labels.py`.
  2. Hardlink N random train + N val images into
     <out_root>/images/{train2017,val2017}/  (RMT-PPAD's split naming
     convention - matches our BDD_lane_only.yaml and P1's lane_targets
     split names).
  3. Extract matching YOLO detection labels (.txt) for those stems from
     `BDD_detection_labels.zip` into <out_root>/labels/{train2017,val2017}/.
  4. Extract matching .pt lane targets from `lane_targets_clr_v1_polyline.tar.gz`
     into <out_root>/lane_targets/{train2017,val2017}/.

Why use the pre-prepared curve tarball instead of `bdd100k_images_100k.zip`?
That tarball is the artifact NB00 produces and NB79 cell 4 uses for P0 -
its layout is fixed and known. The raw images zip's layout varies across
BDD releases.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_split_image_dir(root: Path, split: str) -> Optional[Path]:
    """Find a directory under `root` named after the split that contains
    .jpg files. Handles {train, train2017, training} variants."""
    aliases = {
        'train': ('train', 'train2017', 'training'),
        'val':   ('val', 'val2017', 'validation'),
    }.get(split, (split,))
    for cand in aliases:
        for p in root.rglob(cand):
            if not p.is_dir():
                continue
            n_jpg = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() == '.jpg')
            if n_jpg > 0:
                return p
    return None


def _extract_tar_once(tar_path: Path, dest: Path, marker_name: str) -> None:
    marker = dest / f'.extracted_{marker_name}.ok'
    if marker.exists():
        print(f'  already extracted: {tar_path.name} -> {dest}', flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f'  extracting {tar_path.name} -> {dest}', flush=True)
    with tarfile.open(tar_path, 'r:*') as tf:
        tf.extractall(dest)
    marker.write_text(str(tar_path), encoding='utf-8')
    print(f'  done in {time.time() - t0:.1f}s', flush=True)


def prepare(
    out_root: Path,
    curve_tar: Path,
    labels_zip: Path,
    lane_tar: Path,
    n_train: int,
    n_val: int,
    seed: int,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. Extract the curve tarball (images source).
    print('[prep] step 1: extract curve tarball')
    curve_scratch = Path('/content/bdd_curve_scratch')
    _extract_tar_once(curve_tar, curve_scratch, marker_name=curve_tar.stem)

    train_src = _find_split_image_dir(curve_scratch, 'train')
    val_src = _find_split_image_dir(curve_scratch, 'val')
    if train_src is None or val_src is None:
        raise FileNotFoundError(
            f'Could not find train/val image dirs under {curve_scratch}. '
            f'tar may have an unexpected layout.'
        )
    print(f'  curve images: train={train_src} ({sum(1 for _ in train_src.glob("*.jpg"))} jpgs), '
          f'val={val_src} ({sum(1 for _ in val_src.glob("*.jpg"))} jpgs)')

    # ---- 2. Hardlink subset into out_root/images/{train2017,val2017}/
    print('[prep] step 2: hardlink image subset')
    rng = random.Random(seed)
    all_train_stems = sorted(p.stem for p in train_src.glob('*.jpg'))
    all_val_stems = sorted(p.stem for p in val_src.glob('*.jpg'))
    train_stems = rng.sample(all_train_stems, min(n_train, len(all_train_stems)))
    val_stems = rng.sample(all_val_stems, min(n_val, len(all_val_stems)))
    print(f'  sampled {len(train_stems)} train + {len(val_stems)} val stems')

    img_train_dst = out_root / 'images' / 'train2017'
    img_val_dst   = out_root / 'images' / 'val2017'
    img_train_dst.mkdir(parents=True, exist_ok=True)
    img_val_dst.mkdir(parents=True, exist_ok=True)
    for stem in train_stems:
        _hardlink_or_copy(train_src / f'{stem}.jpg', img_train_dst / f'{stem}.jpg')
    for stem in val_stems:
        _hardlink_or_copy(val_src / f'{stem}.jpg', img_val_dst / f'{stem}.jpg')

    # Re-scan in case some hardlinks dropped (cross-fs etc).
    final_train_stems = sorted(p.stem for p in img_train_dst.glob('*.jpg'))
    final_val_stems = sorted(p.stem for p in img_val_dst.glob('*.jpg'))
    print(f'  on-disk: {len(final_train_stems)} train, {len(final_val_stems)} val')

    # ---- 3. Pull matching detection labels (.txt) from the labels zip.
    print('[prep] step 3: extract matching detection labels')
    want = set(final_train_stems) | set(final_val_stems)
    train_set = set(final_train_stems)
    val_set = set(final_val_stems)
    lbl_train_dst = out_root / 'labels' / 'train2017'
    lbl_val_dst   = out_root / 'labels' / 'val2017'
    lbl_train_dst.mkdir(parents=True, exist_ok=True)
    lbl_val_dst.mkdir(parents=True, exist_ok=True)

    n_train_lbl = n_val_lbl = 0
    with zipfile.ZipFile(labels_zip) as zf:
        all_names = zf.namelist()
        # Probe to make any layout surprise visible.
        print(f'  labels zip: {len(all_names)} entries; first 3: {all_names[:3]}')
        for n in all_names:
            if not n.lower().endswith('.txt'):
                continue
            stem = Path(n).stem
            if stem not in want:
                continue
            data = zf.read(n)
            if stem in train_set:
                out_path = lbl_train_dst / f'{stem}.txt'
            elif stem in val_set:
                out_path = lbl_val_dst / f'{stem}.txt'
            else:
                continue
            if not out_path.exists():
                out_path.write_bytes(data)
                if stem in train_set:
                    n_train_lbl += 1
                else:
                    n_val_lbl += 1
    print(f'  extracted labels: train={n_train_lbl} val={n_val_lbl}')

    # For any image without a label file, create an empty .txt so the YOLO
    # dataset doesn't drop it.
    for stem in final_train_stems:
        p = lbl_train_dst / f'{stem}.txt'
        if not p.exists():
            p.write_text('', encoding='utf-8')
    for stem in final_val_stems:
        p = lbl_val_dst / f'{stem}.txt'
        if not p.exists():
            p.write_text('', encoding='utf-8')

    # ---- 4. Pull matching .pt lane targets from the lane_targets tarball.
    print('[prep] step 4: extract matching lane_targets .pt files')
    lt_train_dst = out_root / 'lane_targets' / 'train2017'
    lt_val_dst   = out_root / 'lane_targets' / 'val2017'
    lt_train_dst.mkdir(parents=True, exist_ok=True)
    lt_val_dst.mkdir(parents=True, exist_ok=True)

    n_train_pt = n_val_pt = 0
    with tarfile.open(lane_tar, 'r:gz') as tf:
        first = []
        for m in tf:
            if len(first) < 3 and m.isfile():
                first.append(m.name)
            if not (m.isfile() and m.name.endswith('.pt')):
                continue
            stem = Path(m.name).stem
            if stem in train_set:
                out_path = lt_train_dst / f'{stem}.pt'
            elif stem in val_set:
                out_path = lt_val_dst / f'{stem}.pt'
            else:
                continue
            if out_path.exists():
                continue
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            out_path.write_bytes(fobj.read())
            if stem in train_set:
                n_train_pt += 1
            else:
                n_val_pt += 1
        if first:
            print(f'  lane tar first entries: {first}')
    print(f'  extracted lane targets: train={n_train_pt} val={n_val_pt}')

    # ---- Final inventory.
    print('\n[prep] final inventory:')
    counts = {}
    for label, d in (
        ('images/train2017',       img_train_dst),
        ('images/val2017',         img_val_dst),
        ('labels/train2017',       lbl_train_dst),
        ('labels/val2017',         lbl_val_dst),
        ('lane_targets/train2017', lt_train_dst),
        ('lane_targets/val2017',   lt_val_dst),
    ):
        n = sum(1 for _ in d.iterdir() if _.is_file())
        counts[label] = n
        print(f'  {label:30s} {n}')

    return {
        'images':       {'train2017': counts['images/train2017'],       'val2017': counts['images/val2017']},
        'labels':       {'train2017': counts['labels/train2017'],       'val2017': counts['labels/val2017']},
        'lane_targets': {'train2017': counts['lane_targets/train2017'], 'val2017': counts['lane_targets/val2017']},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', type=Path, default=Path('/content/bdd_dataset'))
    p.add_argument('--curve-tar', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'),
                   help='Pre-prepared images+labels tarball (NB00 output).')
    p.add_argument('--labels-zip', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/BDD_detection_labels.zip'),
                   help='YOLO-format detection labels (P0 download).')
    p.add_argument('--lane-tar', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/datasets/lane_targets_clr_v1_polyline.tar.gz'),
                   help='Per-image (max_lanes, 78) targets (P1 output).')
    p.add_argument('--n-train', type=int, default=200)
    p.add_argument('--n-val', type=int, default=80)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    for label, path in (
        ('curve_tar',  args.curve_tar),
        ('labels_zip', args.labels_zip),
        ('lane_tar',   args.lane_tar),
    ):
        if not path.exists():
            raise FileNotFoundError(f'Required input missing: {label} = {path}')
        print(f'  {label:11s} -> {path}')

    summary = prepare(
        out_root=args.out_root,
        curve_tar=args.curve_tar,
        labels_zip=args.labels_zip,
        lane_tar=args.lane_tar,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )

    summary_path = args.out_root / 'prep_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\n[prep] summary -> {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
