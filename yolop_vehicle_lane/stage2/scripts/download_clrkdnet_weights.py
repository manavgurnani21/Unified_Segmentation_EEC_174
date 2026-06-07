"""Download CLRKDNet's published checkpoints from GitHub releases.

We use these as a real KD teacher (vs NB72 which self-distilled from a model
of comparable quality and got nothing). CLRKDNet's CULane numbers:
  - ResNet-18 backbone: 79.66 F1 on CULane (small/fast variant)
  - DLA-34 backbone:    80.71 F1 on CULane (paper main result)
  - DLA-34 CLRNet re-run: 80.87 F1 (provided separately by authors)

Both files are hosted at github.com/weiqingq/CLRKDNet/releases. The archive
names below were collected from the README on 2026-05-13.

Usage:
    python yolop_vehicle_lane/stage2/scripts/download_clrkdnet_weights.py \
        --dest /content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights

The script verifies download sizes and exits cleanly if files already exist
(safe to re-run; idempotent).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen, Request


CHECKPOINTS = [
    # (release_url, local_filename, expected_min_bytes, description)
    (
        'https://github.com/weiqingq/CLRKDNet/releases/download/training_logs/dla34_8087.pth',
        'dla34_clrnet_culane_8087.pth',
        50_000_000,
        'CLRNet DLA-34 re-run by authors, 80.87 F1 on CULane '
        '(used as the strong teacher in CLRKDNet paper)',
    ),
    # CLRKDNet has additional release URLs we collected from the README. The
    # ResNet-18 distilled student is reachable from the same releases page.
    (
        'https://github.com/weiqingq/CLRKDNet/releases/download/training_logs/resnet18_distill_log.txt',
        'resnet18_distill_log.txt',
        1_000,
        'Training log of the distilled ResNet-18 student (79.66 F1) -- text '
        'only, useful for debugging our KD path.',
    ),
    (
        'https://github.com/weiqingq/CLRKDNet/releases/download/training_logs/dla34_distillation_log.txt',
        'dla34_distillation_log.txt',
        1_000,
        'Training log of the distilled DLA-34 student (80.68 F1) -- text '
        'only.',
    ),
    (
        'https://github.com/weiqingq/CLRKDNet/releases/download/training_logs/dla_CLRNet_rerun_log.txt',
        'dla_CLRNet_rerun_log.txt',
        1_000,
        'Training log for the dla34_8087.pth teacher run.',
    ),
]


def _download(url: str, dest: Path, expected_min: int) -> None:
    if dest.exists() and dest.stat().st_size >= expected_min:
        print(f'[skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f'[get ] {url}\n       -> {dest}')
    req = Request(url, headers={'User-Agent': 'ecocar-clrkdnet-fetch/1.0'})
    with urlopen(req, timeout=300) as response, open(tmp, 'wb') as out:
        total_bytes = 0
        last_print = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total_bytes += len(chunk)
            if total_bytes - last_print > 25 * (1 << 20):
                print(f'       ... {total_bytes/1e6:.1f} MB')
                last_print = total_bytes
    tmp.rename(dest)
    actual = dest.stat().st_size
    if actual < expected_min:
        raise RuntimeError(
            f'{dest.name} download size {actual} < expected_min {expected_min}; '
            'file may be a redirect HTML page. Check the URL.')
    print(f'[done] {dest.name}  ({actual/1e6:.1f} MB)')


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dest', required=True,
                   help='Destination directory (e.g. '
                        '/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights)')
    args = p.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f'[fetch] CLRKDNet weights into {dest}')
    for url, name, min_bytes, desc in CHECKPOINTS:
        local = dest / name
        try:
            _download(url, local, min_bytes)
            print(f'         {desc}')
        except Exception as e:
            print(f'[fail] {name}: {e}')
    # Verify weight file (sha-style sanity check by just reading head bytes).
    weight = dest / 'dla34_clrnet_culane_8087.pth'
    if weight.exists():
        try:
            import torch  # noqa: WPS433  (import only here so script is importable in CI)
            sd = torch.load(weight, map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            elif isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            keys = list(sd.keys()) if isinstance(sd, dict) else []
            print(f'[verify] {weight.name}: top-level type {type(sd).__name__}, '
                  f'{len(keys)} parameter keys')
            if keys:
                print(f'         first keys: {keys[:5]}')
                print(f'         last keys: {keys[-3:]}')
        except Exception as e:
            print(f'[verify] could not torch.load {weight.name}: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
