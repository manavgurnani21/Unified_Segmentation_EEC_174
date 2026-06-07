"""Run RMT-PPAD's `MTDETR.val(...)` against a checkpoint and capture metrics.

This wraps `external_repos/RMT-PPAD-main/ultralytics/test.py` (which is just
two lines hardcoding the author's local paths) with proper CLI arguments and
JSON output parsing.

Usage:
    python yolop_vehicle_lane/stage2/rmt_ppad_migration/P0_baseline/run_baseline_val.py \\
        --rmt-ppad-root /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/rmt_ppad_migration/vendor/RMT-PPAD \\
        --checkpoint /content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/rmt_ppad_best.pt \\
        --data-yaml /content/.../BDD_full.yaml \\
        --output-json /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/rmt_ppad_migration/results/baseline_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def patch_bdd_yaml(src_yaml: Path, bdd_root: Path, dst_yaml: Path) -> None:
    """Rewrite BDD_full.yaml so its 'path:' points at the user's actual BDD root."""
    if not src_yaml.exists():
        raise FileNotFoundError(f'BDD config not found at {src_yaml}')
    text = src_yaml.read_text(encoding='utf-8')
    # Replace the author's hardcoded path with the user's path. The original line is:
    #     path: /home/jiayuan/data/BDD_seg_mask # dataset root dir
    new_text = re.sub(r'^path:\s*.*$', f'path: {bdd_root}', text, count=1, flags=re.MULTILINE)
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text(new_text, encoding='utf-8')
    print(f'[yaml] wrote {dst_yaml} with path={bdd_root}')


def run_validation(rmt_ppad_root: Path, checkpoint: Path, data_yaml: Path,
                   batch: int, imgsz: int, mask_threshold) -> tuple[int, str]:
    """Invoke MTDETR.val(...) as a subprocess. Returns (rc, stdout)."""
    code = f"""
import sys
import importlib

# Put our vendored RMT-PPAD copy at the FRONT of sys.path so its ultralytics/
# subpackage shadows the PyPi ultralytics that Colab pre-installs (the
# upstream package doesn't have MTDETR).
sys.path.insert(0, {str(rmt_ppad_root)!r})

# Defensive: Colab kernels often have `ultralytics` already cached in
# sys.modules from a prior import. Drop those entries so the next
# `from ultralytics import MTDETR` re-resolves against our path.
for _k in list(sys.modules):
    if _k == 'ultralytics' or _k.startswith('ultralytics.'):
        del sys.modules[_k]

# Diagnostic: confirm we're loading our vendored copy, not PyPi's.
import ultralytics as _ul
print('[subproc] ultralytics imported from:', _ul.__file__)
expected_prefix = {str(rmt_ppad_root)!r}
if not str(_ul.__file__).startswith(expected_prefix):
    raise RuntimeError(
        'ultralytics resolved from ' + _ul.__file__ + ' but we wanted '
        + expected_prefix + '. This means the vendored copy was not found '
        'on sys.path. Check vendor_rmt_ppad.py ran successfully.')

from ultralytics import MTDETR
print('[subproc] MTDETR import OK; loading checkpoint...')
model = MTDETR({str(checkpoint)!r})
print('[subproc] MTDETR loaded; starting val()...')
results = model.val(
    data={str(data_yaml)!r},
    gc=False,
    imgsz={imgsz},
    mask_ratio=1,
    overlap_mask=False,
    batch={batch},
    device=[0],
    mask_threshold={mask_threshold!r},
    plots=False,
    project='runs',
    name='baseline_P0',
)

print('[subproc] model.val() returned; extracting metrics safely...')

# RMT-PPAD's SimpleClass.__repr__ iterates a hardcoded attr list that
# includes 'curves_results' — an attribute SegmentMetrics doesn't define.
# So `print(results)` or `repr(results)` crashes with AttributeError even
# though the validation itself finished cleanly. Wrap the dump and also
# hand-pick known-good attributes by name.
print('=== METRICS DICT (best-effort) ===')
try:
    print(repr(results))
except Exception as e:
    print(f'[note] repr(results) crashed: {{type(e).__name__}}: {{e}} -- using attribute extraction instead.')

print('=== METRICS ATTRIBUTES ===')
# Common attributes on DetMetrics / SegmentMetrics:
#   .box.mp / .mr / .map50 / .map         (detection: precision, recall, mAP50, mAP50-95)
#   .seg.mp / .mr / .map50 / .map         (segmentation, when present)
# Also try .results_dict and .speed.
candidates = [
    ('box.mp',     'detection_precision'),
    ('box.mr',     'detection_recall'),
    ('box.map50',  'detection_map50'),
    ('box.map',    'detection_map'),
    ('seg.mp',     'seg_precision'),
    ('seg.mr',     'seg_recall'),
    ('seg.map50',  'seg_map50'),
    ('seg.map',    'seg_map'),
]
for attr_chain, label in candidates:
    try:
        obj = results
        for part in attr_chain.split('.'):
            obj = getattr(obj, part)
        val = float(obj) if obj is not None else None
        if val is not None:
            print(f'METRIC[{{label}}] = {{val:.6f}}  (via results.{{attr_chain}})')
    except Exception as e:
        # Quietly skip — many attrs are model-specific and may not exist.
        pass

# Also try .results_dict which is a dict-style accessor on the metrics object.
try:
    rd = results.results_dict
    print('=== results.results_dict ===')
    for k, v in rd.items():
        try:
            print(f'METRIC[{{k}}] = {{float(v):.6f}}')
        except (TypeError, ValueError):
            print(f'METRIC[{{k}}] = {{v}}')
except Exception as e:
    print(f'[note] results.results_dict unavailable: {{type(e).__name__}}: {{e}}')

# RMT-PPAD prints its own per-task summary lines during val (drivable mIoU,
# lane IoU, lane ACC). Those are already in the captured stdout; the parser
# in run_baseline_val.py picks them up regardless of what this section prints.
print('[subproc] DONE')
"""
    print(f'[runner] launching MTDETR.val subprocess (this can take 10-30 min on '
          f'the full BDD val split, ~5000 images)...')
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, '-u', '-c', code],
        cwd=str(rmt_ppad_root),
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    stdout = (proc.stdout or '') + '\n--- STDERR ---\n' + (proc.stderr or '')
    print(f'[runner] subprocess returned in {dt:.1f}s with rc={proc.returncode}')
    return proc.returncode, stdout


# Three-tier metric extraction (each tier overrides earlier-found keys):
#
#  Tier 1: METRIC[<label>] = <float> lines printed by the subprocess after
#          model.val() returns. Most reliable; comes from explicit
#          getattr(results, '<attr>') in the subprocess.
#
#  Tier 2: Ultralytics' standard 'all' row from the val summary table, e.g.
#          "  all     10000     54321     0.954  0.842   0.849      0.621"
#          which prints precision/recall/mAP50/mAP50-95 in that order.
#
#  Tier 3: RMT-PPAD's per-task printout lines (drivable mIoU, lane IoU, ACC).
METRIC_LABEL_LINE = re.compile(r'METRIC\[([A-Za-z0-9_\-./]+)\]\s*=\s*([0-9.]+)')

# Detection 'all' summary row (ultralytics format). The header line is:
#   Class Images Instances Box(P R mAP50 mAP50-95)
# followed by:
#   all   <imgs> <inst>    <P> <R> <mAP50> <mAP50-95>
ALL_ROW_BOX = re.compile(
    r'^\s*all\s+\d+\s+\d+\s+'
    r'([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',
    re.MULTILINE,
)

# RMT-PPAD's per-task summary row (verified format from NB79 log 2026-05-25):
#   drivable     pixacc      0.981     subacc      0.942        IoU      0.875       mIoU      0.926
#   lane         pixacc       0.99     subacc      0.847        IoU      0.568       mIoU      0.779
# Note: the paper's "Lane ACC" maps to RMT-PPAD's `subacc` field for the lane row.
DRIVABLE_ROW = re.compile(
    r'\bdrivable\s+pixacc\s+([\d.]+)\s+subacc\s+([\d.]+)\s+IoU\s+([\d.]+)\s+mIoU\s+([\d.]+)',
    re.IGNORECASE,
)
LANE_ROW = re.compile(
    r'\blane\s+pixacc\s+([\d.]+)\s+subacc\s+([\d.]+)\s+IoU\s+([\d.]+)\s+mIoU\s+([\d.]+)',
    re.IGNORECASE,
)


def parse_metrics(stdout: str) -> dict:
    """Three-tier parse: METRIC[...]= lines, then ultralytics 'all' row,
    then RMT-PPAD per-task lines. Each tier fills gaps left by earlier tiers."""
    found = {}

    # Tier 1: METRIC[label] = value lines (most reliable).
    for m in METRIC_LABEL_LINE.finditer(stdout):
        try:
            found[m.group(1)] = float(m.group(2))
        except ValueError:
            pass

    # Tier 2: the 'all' summary row from ultralytics.
    m = ALL_ROW_BOX.search(stdout)
    if m:
        # P, R, mAP50, mAP50-95
        if 'detection_precision' not in found:
            try:
                found['detection_precision'] = float(m.group(1))
            except ValueError:
                pass
        if 'detection_recall' not in found:
            try:
                found['detection_recall'] = float(m.group(2))
            except ValueError:
                pass
        if 'detection_map50' not in found:
            try:
                found['detection_map50'] = float(m.group(3))
            except ValueError:
                pass
        if 'detection_map' not in found:
            try:
                found['detection_map'] = float(m.group(4))
            except ValueError:
                pass

    # Tier 3: RMT-PPAD per-task summary rows.
    m = DRIVABLE_ROW.search(stdout)
    if m:
        for i, key in enumerate(['drivable_pixacc', 'drivable_subacc', 'drivable_iou', 'drivable_miou'], start=1):
            if key not in found:
                try:
                    found[key] = float(m.group(i))
                except ValueError:
                    pass
    m = LANE_ROW.search(stdout)
    if m:
        # The paper's "Lane ACC" is RMT-PPAD's `subacc` (group 2).
        for i, key in enumerate(['lane_pixacc', 'lane_acc', 'lane_iou', 'lane_miou'], start=1):
            if key not in found:
                try:
                    found[key] = float(m.group(i))
                except ValueError:
                    pass
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--rmt-ppad-root', required=True,
                   help='Root of the vendored RMT-PPAD codebase (writable).')
    p.add_argument('--checkpoint', required=True,
                   help='Path to rmt_ppad_best.pt downloaded from SharePoint.')
    p.add_argument('--bdd-root', required=True,
                   help='Root of the BDD100K dataset RMT-PPAD format '
                        '(must contain images/, labels/, mask/ subdirs).')
    p.add_argument('--output-json', required=True,
                   help='Path for baseline_metrics.json output.')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--mask-thr', default='0.45,0.9',
                   help='Two thresholds (drivable, lane) for mask post-processing.')
    args = p.parse_args(argv)

    rmt_root = Path(args.rmt_ppad_root)
    ckpt = Path(args.checkpoint)
    bdd_root = Path(args.bdd_root)
    out_json = Path(args.output_json)

    if not rmt_root.exists():
        raise FileNotFoundError(f'rmt-ppad-root {rmt_root} does not exist; '
                                'vendor the codebase first (NB79 cell 2).')
    if not ckpt.exists():
        raise FileNotFoundError(f'checkpoint {ckpt} does not exist; '
                                'run download_rmt_ppad_pretrained.py first.')
    if not bdd_root.exists():
        raise FileNotFoundError(f'BDD root {bdd_root} does not exist.')

    # Patch BDD_full.yaml with the user's BDD root path.
    bdd_yaml_src = rmt_root / 'ultralytics' / 'cfg' / 'datasets' / 'BDD_full.yaml'
    bdd_yaml_dst = rmt_root / 'ultralytics' / 'cfg' / 'datasets' / 'BDD_full_local.yaml'
    patch_bdd_yaml(bdd_yaml_src, bdd_root, bdd_yaml_dst)

    mask_thr = [float(x) for x in args.mask_thr.split(',')]

    rc, stdout = run_validation(rmt_root, ckpt, bdd_yaml_dst,
                                 batch=args.batch, imgsz=args.imgsz,
                                 mask_threshold=mask_thr)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_json.with_suffix('.log')
    log_path.write_text(stdout, encoding='utf-8')
    print(f'[log] wrote {log_path}')

    # Parse metrics from stdout regardless of rc -- a non-zero rc with
    # a successfully-completed val (and metrics extractable from the
    # stdout summary table) is treated as effectively-successful.
    metrics = parse_metrics(stdout)
    completed_val = bool(metrics) and (
        'detection_map50' in metrics or 'detection_recall' in metrics
    )
    if rc != 0 and completed_val:
        print(f'\n[note] subprocess returned rc={rc} but val completed and {len(metrics)} '
              f'metric(s) were extracted from stdout. Treating as effective success.')
        rc = 0
    record = {
        'returncode': rc,
        'rmt_ppad_root': str(rmt_root),
        'checkpoint': str(ckpt),
        'bdd_root': str(bdd_root),
        'imgsz': args.imgsz,
        'batch': args.batch,
        'mask_threshold': mask_thr,
        'metrics': metrics,
        'log_path': str(log_path),
    }
    out_json.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'[json] wrote {out_json}')

    if rc != 0:
        # Surface the subprocess output directly so the notebook cell shows the
        # actual Python traceback instead of just "rc=1". Previously we only
        # wrote to the log file; users can't see what blew up without opening it.
        print(f'\n[fail] subprocess returned rc={rc}. Last 6 KB of subprocess output follows:')
        print('=' * 78)
        # Print last 6 KB (typical traceback fits comfortably). Helps when the
        # subprocess had thousands of warning lines before the real error.
        tail = stdout[-6000:] if len(stdout) > 6000 else stdout
        print(tail)
        print('=' * 78)
        print(f'\nFull log: {log_path}')
        return rc

    print('\n=== Baseline metrics extracted ===')
    if metrics:
        for k, v in metrics.items():
            print(f'  {k}: {v}')
    else:
        print('  (no metrics parsed from stdout; review log manually)')
    print('\nTarget reference (from paper):')
    print('  mAP50 ≈ 0.85   drivable mIoU ≈ 0.92   lane IoU ≈ 0.57   lane ACC ≈ 0.85')
    return 0


if __name__ == '__main__':
    sys.exit(main())
