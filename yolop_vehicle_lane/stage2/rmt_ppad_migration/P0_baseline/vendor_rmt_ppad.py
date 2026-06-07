"""Shallow-vendor RMT-PPAD source code into stage2/rmt_ppad_migration/vendor/RMT-PPAD/.

Skips build artifacts (build/, ultralytics.egg-info/, docs/, examples/, tests/,
pictures/, large PDFs) so the vendor folder stays manageable in Drive.

We need a WRITABLE copy because P2-P8 will modify files. The original
external_repos/RMT-PPAD-main/ stays untouched (rule from the migration plan).

Idempotent: re-runs only copy files newer than the destination.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# Top-level files/dirs to copy. Anything else is skipped.
KEEP_DIRS = ['ultralytics']
KEEP_FILES = ['README.md', 'environment.yml', 'pyproject.toml', 'LICENSE',
              'CITATION.cff', 'CONTRIBUTING.md']
# Within ultralytics/, skip these (they're auto-generated or test artifacts).
SKIP_SUBDIRS = {'__pycache__', '.egg-info'}


def _should_skip(name: str) -> bool:
    if name in SKIP_SUBDIRS:
        return True
    if name.endswith('.egg-info'):
        return True
    if name.startswith('.'):
        return True
    return False


def _copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    """Copy `src` directory to `dst`, skipping junk. Returns (files_copied, bytes_copied)."""
    n_files = 0
    n_bytes = 0
    for root, dirs, files in __import__('os').walk(src):
        # In-place prune of dirs based on skip rules
        dirs[:] = [d for d in dirs if not _should_skip(d)]
        rel = Path(root).relative_to(src)
        for f in files:
            if _should_skip(f) or f.endswith('.pyc'):
                continue
            sp = Path(root) / f
            dp = dst / rel / f
            dp.parent.mkdir(parents=True, exist_ok=True)
            if dp.exists() and dp.stat().st_mtime >= sp.stat().st_mtime and dp.stat().st_size == sp.stat().st_size:
                continue  # idempotent skip
            shutil.copy2(sp, dp)
            n_files += 1
            n_bytes += sp.stat().st_size
    return n_files, n_bytes


def vendor(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f'source not found: {src_root}')
    dst_root.mkdir(parents=True, exist_ok=True)

    # Top-level files
    for fname in KEEP_FILES:
        sp = src_root / fname
        if sp.exists():
            dp = dst_root / fname
            if not dp.exists() or dp.stat().st_mtime < sp.stat().st_mtime:
                shutil.copy2(sp, dp)
                print(f'[file] {fname} ({sp.stat().st_size/1024:.1f} KB)')

    # Top-level directories (currently just ultralytics/)
    total_files = 0
    total_bytes = 0
    for dname in KEEP_DIRS:
        sd = src_root / dname
        dd = dst_root / dname
        if not sd.exists():
            print(f'[skip] {dname} not found in source')
            continue
        n_files, n_bytes = _copy_tree(sd, dd)
        total_files += n_files
        total_bytes += n_bytes
        print(f'[dir]  {dname}: {n_files} files, {n_bytes/1e6:.1f} MB')

    print(f'\n[done] {total_files} files copied / updated, {total_bytes/1e6:.1f} MB')
    print(f'       vendored at: {dst_root}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True,
                   help='Source: external_repos/RMT-PPAD-main/ (read-only reference).')
    p.add_argument('--dst', required=True,
                   help='Destination: stage2/rmt_ppad_migration/vendor/RMT-PPAD/ '
                        '(writable working copy).')
    args = p.parse_args(argv)
    vendor(Path(args.src), Path(args.dst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
