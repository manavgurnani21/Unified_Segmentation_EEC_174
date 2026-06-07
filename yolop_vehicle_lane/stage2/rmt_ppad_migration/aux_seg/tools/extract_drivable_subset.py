"""Extract BDD drivable masks for the 10k subset stems from a seg-maps zip.

The aux drivable head needs a per-image dense GT mask. BDD ships these inside
`bdd100k_seg_maps.zip`; we pull only the masks whose stem matches a subset
image and write a binary drivable mask per stem to
    <subset>/drivable_masks/{train2017,val2017}/<stem>.png

The zip's exact internal layout varies by BDD release (drivable_id vs seg
train_id, nested folders, name suffixes), so this script is SELF-INSPECTING:
it prints the structure + a value histogram and matches by normalized stem.
By default any non-zero pixel = drivable foreground (correct for BDD
drivable_id masks where 1/2 are the drivable areas; for semantic-seg maps it
yields a coarse annotated-region mask, still a valid dense aux signal). Pass
--road-ids to keep only specific class ids instead.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

_SUFFIXES = ('_drivable_id', '_drivable_color', '_train_id', '_id', '_drivable',
             '_seg', '_color', '_labelIds')


def _norm_stem(name: str) -> str:
    stem = Path(name).stem
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-zip', type=Path, required=True)
    ap.add_argument('--subset-root', type=Path, default=Path('/content/bdd_subset_10k'))
    ap.add_argument('--out-subdir', default='drivable_masks')
    ap.add_argument('--splits', nargs='+', default=['train2017', 'val2017'])
    ap.add_argument('--road-ids', nargs='*', type=int, default=None,
                    help='If set, drivable = pixel in these class ids. Else >0.')
    ap.add_argument('--inspect-only', action='store_true',
                    help='Just print the zip structure + a value histogram and exit.')
    args = ap.parse_args()

    if not args.seg_zip.is_file():
        print(f'FATAL: seg zip not found: {args.seg_zip}', flush=True)
        return 2

    zf = zipfile.ZipFile(args.seg_zip)
    png_entries = [n for n in zf.namelist() if n.lower().endswith('.png')]
    print(f'[inspect] {args.seg_zip.name}: {len(zf.namelist())} entries, '
          f'{len(png_entries)} png', flush=True)
    for n in png_entries[:8]:
        print(f'   e.g. {n}')
    if png_entries:
        sample = cv2.imdecode(np.frombuffer(zf.read(png_entries[0]), np.uint8),
                              cv2.IMREAD_GRAYSCALE)
        if sample is not None:
            vals, cnts = np.unique(sample, return_counts=True)
            print(f'[inspect] first mask {png_entries[0]} unique values: '
                  f'{dict(zip(vals.tolist(), cnts.tolist()))}')
    if args.inspect_only:
        return 0

    # stem -> zip entry (prefer entries that look split-correct later)
    index: dict[str, list[str]] = {}
    for n in png_entries:
        index.setdefault(_norm_stem(n), []).append(n)

    total_done = total_missing = 0
    for split in args.splits:
        img_dir = args.subset_root / 'images' / split
        if not img_dir.is_dir():
            print(f'[skip] no images at {img_dir}')
            continue
        out_dir = args.subset_root / args.out_subdir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = [p.stem for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')]
        done = missing = 0
        for st in stems:
            cands = index.get(st)
            # try a looser match (split hint) if exact normalized stem missed
            if not cands:
                cands = index.get(_norm_stem(st))
            if not cands:
                missing += 1
                continue
            # Rank candidates. The real BDD drivable label set ships BOTH an id
            # mask (labels/drivable/masks/<split>/<stem>.png, values 0/1/2 - the
            # dense GT we want) AND an RGB colormap (.../colormaps/...). Prefer
            # the split match, the id 'masks/' dir, and penalize 'color' so a
            # colormap is never chosen when an id mask exists for the stem.
            def _rank(c):
                cl = c.lower()
                s = 0
                if split in cl or split[:-4] in cl:
                    s += 4
                if 'mask' in cl:
                    s += 2
                if 'drivable' in cl:
                    s += 1
                if 'color' in cl:
                    s -= 3
                return s
            entry = max(cands, key=_rank)
            raw = cv2.imdecode(np.frombuffer(zf.read(entry), np.uint8), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                missing += 1
                continue
            if args.road_ids:
                drivable = np.isin(raw, args.road_ids).astype(np.uint8)
            else:
                drivable = (raw > 0).astype(np.uint8)
            cv2.imwrite(str(out_dir / f'{st}.png'), drivable)
            done += 1
        print(f'[{split}] wrote {done} drivable masks, {missing} missing '
              f'(of {len(stems)} images)', flush=True)
        total_done += done
        total_missing += missing

    print(f'\n[done] {total_done} masks written, {total_missing} missing -> '
          f'{args.subset_root / args.out_subdir}', flush=True)
    if total_done == 0:
        print('[WARN] 0 masks matched - the stem/suffix scheme differs. Re-run '
              'with --inspect-only and report the printed entry names.', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
