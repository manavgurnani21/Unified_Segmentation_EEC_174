"""Extract CULane archives into the structure BDDJointCurveDataset expects.

CULane's native layout is already what we read (`list/train_gt.txt`,
`.lines.txt` next to each image), so this script just unpacks the archives
into one directory and verifies counts.

Expected source layout (from user's Drive):
    /content/drive/MyDrive/EcoCAR/downloads/CULane/
        list.tar.gz
        annotations_new.tar.gz
        driver_23_30frame.tar.gz   (or any subset of the 6 driver archives)
        driver_37_30frame.tar.gz
        ...

Target layout:
    /content/CULane/
        list/train_gt.txt
        list/val.txt
        driver_<X>_<Y>frame/<sequence>/*.jpg
        driver_<X>_<Y>frame/<sequence>/*.lines.txt

The function works incrementally -- skips archives that are already extracted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DRIVER_ARCHIVES = [
    'driver_23_30frame.tar.gz',
    'driver_37_30frame.tar.gz',
    'driver_100_30frame.tar.gz',
    'driver_161_90frame.tar.gz',
    'driver_182_30frame.tar.gz',
    'driver_193_90frame.tar.gz',
]

# Approximate jpg counts per driver from CULane's published statistics. Used as
# a "good enough" threshold for marker validity in `_extract`. If the on-disk
# count is at least 90 % of expected, we skip re-extraction; otherwise we
# treat the directory as partial and re-extract.
EXPECTED_JPGS = {
    'driver_23_30frame': 88000,    # 20.44 GB archive
    'driver_37_30frame': 5400,     # 1.04 GB
    'driver_100_30frame': 22000,   # 4.75 GB
    'driver_161_90frame': 22000,   # 4.69 GB
    'driver_182_30frame': 27000,   # 5.49 GB
    'driver_193_90frame': 20000,   # 4.20 GB
}


def _count_jpg(directory: Path, cap: int = 200_000) -> int:
    """Count .jpg files under `directory` (capped so it doesn't take forever)."""
    n = 0
    for _ in directory.rglob('*.jpg'):
        n += 1
        if n >= cap:
            break
    return n


def _extract(archive: Path, dest: Path, marker_dir: str | None,
             force: bool = False, expected_min_jpgs: int = 0) -> bool:
    """Extract `archive` into `dest`. Skip if marker dir already has the
    expected number of .jpgs (so partial / aborted prior extractions get
    re-run). When `force=True`, always re-extract."""
    if not archive.exists():
        print(f'[missing] {archive} -- skipping')
        return False
    if marker_dir and (dest / marker_dir).exists():
        existing = _count_jpg(dest / marker_dir)
        if not force and existing >= max(1, expected_min_jpgs):
            print(f'[skip-extracted] {archive.name} -> {dest / marker_dir} '
                  f'already has {existing} jpgs (>= expected {expected_min_jpgs})')
            return False
        elif not force and existing > 0 and expected_min_jpgs > 0:
            print(f'[partial-extracted] {archive.name} -> {dest / marker_dir} '
                  f'has only {existing} of expected ~{expected_min_jpgs} jpgs; '
                  f're-extracting on top to fill gaps.')
        # else: continue to extract
    dest.mkdir(parents=True, exist_ok=True)
    print(f'[extract] {archive} -> {dest}')
    if str(archive).endswith('.tar.gz') or str(archive).endswith('.tgz'):
        subprocess.check_call(['tar', '-xzf', str(archive), '-C', str(dest)])
    elif str(archive).endswith('.zip'):
        subprocess.check_call(['unzip', '-q', '-o', str(archive), '-d', str(dest)])
    else:
        subprocess.check_call(['tar', '-xf', str(archive), '-C', str(dest)])
    return True


def prepare(src_dir: str, dest_dir: str, drivers: list[str] | None = None,
            force: bool = False) -> None:
    src = Path(src_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Diagnostic: NB74 v1 silently skipped every archive because the user-
    # provided src dir was empty / under a different path / not yet synced
    # from Drive. List what we actually see so the user can fix the path
    # without re-running the whole notebook.
    if not src.exists():
        raise FileNotFoundError(
            f'src directory {src} does not exist. Verify the path or wait for '
            'Drive to sync. List MyDrive/EcoCAR/downloads/ in Colab to find '
            'where the CULane archives actually are.')
    visible = sorted(src.iterdir())
    print(f'[src] {src} -- {len(visible)} entries:')
    for p in visible:
        try:
            sz = p.stat().st_size
            print(f'  {p.name}  ({sz/1e9:.2f} GB)')
        except OSError:
            print(f'  {p.name}  (size unknown)')
    if not visible:
        # Search one level up for a plausible CULane directory.
        parent = src.parent
        candidates = []
        if parent.exists():
            for p in parent.iterdir():
                if p.is_dir() and any(p.glob('*.tar.gz')):
                    candidates.append(p)
        msg = (f'No files found under {src}. '
               f'Tip: list {parent} or look in your Drive for the CULane archives.')
        if candidates:
            msg += f' Found tar.gz files in: {[str(c) for c in candidates]}'
        raise FileNotFoundError(msg)

    # list.tar.gz unpacks to dest/list/{train_gt.txt, val.txt, test.txt, ...}
    _extract(src / 'list.tar.gz', dest, marker_dir='list', force=force)

    # annotations_new.tar.gz contains the .lines.txt label files. They unpack
    # into the same per-driver subdirectories that the images use.
    _extract(src / 'annotations_new.tar.gz', dest, marker_dir=None, force=force)

    # Driver archives = the actual images. User can pick a subset for fast
    # iteration -- a single driver (~1 GB) is enough for a backbone pretrain
    # at limit=3000 samples, all 6 (~40 GB) for a full CULane run.
    archive_list = drivers if drivers else DRIVER_ARCHIVES
    for arch_name in archive_list:
        # Pick a sample subdir to use as the "already extracted" marker.
        marker = arch_name.replace('.tar.gz', '')
        # 90 % of expected jpgs is "good enough"; below that we re-extract.
        threshold = int(0.9 * EXPECTED_JPGS.get(marker, 1)) if marker in EXPECTED_JPGS else 1
        _extract(src / arch_name, dest, marker_dir=marker,
                 force=force, expected_min_jpgs=threshold)

    # Post-extraction integrity report (per driver). The crash in NB74 v1
    # happened because earlier sessions extracted some drivers only partially.
    print('\n[integrity] per-driver jpg counts (90 % threshold for "complete"):')
    for arch_name in archive_list:
        marker = arch_name.replace('.tar.gz', '')
        actual = _count_jpg(dest / marker) if (dest / marker).exists() else 0
        expected = EXPECTED_JPGS.get(marker, 0)
        ratio = (actual / expected) if expected else float('inf')
        status = 'OK' if (expected and actual >= 0.9 * expected) else ('PARTIAL' if actual > 0 else 'MISSING')
        print(f'  {marker:25s} {actual:7d} / ~{expected:7d}  ({ratio*100:.1f}%) [{status}]')

    # Sanity report.
    list_dir = dest / 'list'
    if list_dir.exists():
        for split in ('train_gt.txt', 'val.txt', 'test.txt'):
            f = list_dir / split
            if f.exists():
                n = sum(1 for _ in f.open('r', encoding='utf-8'))
                print(f'[list] {split}: {n} lines')
            else:
                print(f'[list] {split}: MISSING')
    # Count .lines.txt files we actually have.
    lane_label_count = 0
    for p in dest.rglob('*.lines.txt'):
        lane_label_count += 1
        if lane_label_count >= 200000:
            break
    print(f'[verify] found {lane_label_count} .lines.txt label files under {dest}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Directory containing the CULane archives (e.g. /content/drive/MyDrive/EcoCAR/downloads/CULane).')
    parser.add_argument('--dest', required=True, help='Output directory to extract into (e.g. /content/CULane).')
    parser.add_argument('--drivers', nargs='*', default=None, help='Subset of driver archives to extract. Default: all 6.')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extract everything, even if marker dirs already exist. '
                             'Use this if a prior Colab session crashed mid-extraction.')
    args = parser.parse_args(argv)
    prepare(args.src, args.dest, args.drivers, force=args.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
