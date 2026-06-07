"""Download RMT-PPAD's pretrained best.pt from the SharePoint share link.

The README's SharePoint URL is an "anonymous share" link. Direct GET on it
returns an HTML preview page, not the .pt file. We append `&download=1`
which SharePoint interprets as "binary download" and 302-redirects to the
actual file URL.

Usage (idempotent, safe to re-run):
    python yolop_vehicle_lane/stage2/rmt_ppad_migration/P0_baseline/download_rmt_ppad_pretrained.py \\
        --dest /content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights

After this finishes, the file should be at
`<dest>/rmt_ppad_best.pt` and be roughly 130-300 MB (RMT-PPAD-l checkpoint
size depends on whether the encoder is included).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request


# SharePoint share URLs from RMT-PPAD's README. Each gets `?download=1`
# appended to force binary download instead of HTML preview.
SHARE_URLS = {
    'rmt_ppad_best.pt': (
        'https://uwin365-my.sharepoint.com/:u:/g/personal/wang621_uwindsor_ca/'
        'EVvXPuqxXdRAkIuAVdth14gBYKuDJ6XqlA2ppRHsmeQN_w?e=hKcXJX',
        5 * 1024 * 1024,  # expect 135-280 MB; HTML preview is < 5 MB
        'pretrained model weights',
    ),
    'BDD_detection_labels.zip': (
        'https://uwin365-my.sharepoint.com/:u:/g/personal/wang621_uwindsor_ca/'
        'EV2FyiQg0llNpBL2F5hnEi0BwfEFTP3jckw7adfLSXPzrQ?e=jSaTOO',
        1 * 1024 * 1024,
        'detection labels in YOLO format (RMT-PPAD\'s preprocessed BDD100K)',
    ),
    'BDD_seg_masks.zip': (
        'https://uwin365-my.sharepoint.com/:u:/g/personal/wang621_uwindsor_ca/'
        'EXrUtDWQ5vlAgzaGopIC3foBZXbs5JNNJRgvR4XotO2cgg?e=CVLOHg',
        50 * 1024 * 1024,
        'lane + drivable segmentation masks',
    ),
}

# Default MIN_BYTES if a target isn't in SHARE_URLS (back-compat for older callers).
MIN_BYTES = 5 * 1024 * 1024


def _try_python_urllib(url: str, dest: Path) -> bool:
    """First try: Python urllib. Works only if SharePoint doesn't require JS."""
    tmp = dest.with_suffix(dest.suffix + '.part')
    try:
        req = Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ecocar-rmt-ppad-fetch/1.0',
            'Accept': '*/*',
        })
        print(f'[urllib] GET {url}')
        with urlopen(req, timeout=120) as resp, open(tmp, 'wb') as out:
            n = 0
            last_print = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                n += len(chunk)
                if n - last_print > 25 * (1 << 20):
                    print(f'         ... {n/1e6:.1f} MB')
                    last_print = n
        size = tmp.stat().st_size
        if size < MIN_BYTES:
            print(f'[urllib] result {size} bytes is below threshold {MIN_BYTES}; '
                  'this is likely SharePoint preview HTML, not the .pt file.')
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f'[urllib] failed: {e}')
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _try_curl(url: str, dest: Path) -> bool:
    """Second try: curl with redirect following and a real User-Agent.
    SharePoint anonymous links often work with curl when they reject urllib."""
    if not shutil.which('curl'):
        print('[curl] not installed; skipping curl path.')
        return False
    tmp = dest.with_suffix(dest.suffix + '.part')
    cmd = [
        'curl', '-L', '--fail', '--retry', '3', '--connect-timeout', '60',
        '-A', 'Mozilla/5.0 (X11; Linux x86_64) ecocar-rmt-ppad-fetch/1.0',
        '-o', str(tmp),
        url,
    ]
    print('[curl] ' + ' '.join(cmd))
    try:
        rc = subprocess.call(cmd)
        if rc != 0 or not tmp.exists():
            return False
        size = tmp.stat().st_size
        if size < MIN_BYTES:
            print(f'[curl] {size} bytes is below threshold; probably preview HTML.')
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f'[curl] failed: {e}')
        return False


def _try_wget(url: str, dest: Path) -> bool:
    """Third try: wget. Some Colab kernels have wget but not curl, or vice versa."""
    if not shutil.which('wget'):
        return False
    tmp = dest.with_suffix(dest.suffix + '.part')
    cmd = ['wget', '--no-check-certificate', '-O', str(tmp), url]
    print('[wget] ' + ' '.join(cmd))
    try:
        rc = subprocess.call(cmd)
        if rc != 0 or not tmp.exists():
            return False
        size = tmp.stat().st_size
        if size < MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f'[wget] failed: {e}')
        return False


def download_one(dest_dir: Path, filename: str) -> Path:
    """Download a single file from the SHARE_URLS table."""
    if filename not in SHARE_URLS:
        raise KeyError(f'Unknown target {filename}. Known: {list(SHARE_URLS)}')
    share_url, min_bytes, description = SHARE_URLS[filename]
    download_url = share_url.split('?')[0] + '?download=1'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f'[skip] {dest} already present ({dest.stat().st_size/1e6:.1f} MB)')
        return dest
    print(f'[fetch] {filename} ({description}) -> {dest_dir}')

    # Temporarily override MIN_BYTES check in the helpers via closures by
    # writing to a local _check function -- but the existing helpers use the
    # global MIN_BYTES. Easiest: monkey-patch MIN_BYTES for this download.
    global MIN_BYTES
    saved = MIN_BYTES
    MIN_BYTES = min_bytes
    try:
        for fn in (_try_python_urllib, _try_curl, _try_wget):
            ok = fn(download_url, dest)
            if ok:
                print(f'[done] {dest}  ({dest.stat().st_size/1e6:.1f} MB)')
                return dest
    finally:
        MIN_BYTES = saved

    # All three failed. Print actionable manual fallback.
    raise RuntimeError(
        f'All three download methods failed for {filename}.\n\n'
        'Most common cause: SharePoint anonymous-link policy requires browser '
        'consent. Manual fallback:\n'
        '  1) Open in a browser:\n'
        f'     {share_url}\n'
        '  2) Click "Download" (top-right toolbar).\n'
        f'  3) Save the file as {dest}.\n'
        '  4) Re-run this script; it will see the existing file and skip.\n'
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dest', required=True, help='Destination directory.')
    p.add_argument('--targets', nargs='+',
                   default=list(SHARE_URLS.keys()),
                   choices=list(SHARE_URLS.keys()),
                   help='Which files to fetch. Default: all three.')
    args = p.parse_args(argv)
    dest_dir = Path(args.dest)
    print(f'[plan] downloading {len(args.targets)} target(s) into {dest_dir}')
    written = []
    for name in args.targets:
        try:
            f = download_one(dest_dir, name)
            written.append(str(f))
        except Exception as e:
            print(f'[fail] {name}: {e}')
    print('\n=== Summary ===')
    for f in written:
        print(f'  OK  {f}')
    if len(written) < len(args.targets):
        print(f'\n{len(args.targets) - len(written)} file(s) failed to download '
              f'automatically. See manual fallback messages above.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
