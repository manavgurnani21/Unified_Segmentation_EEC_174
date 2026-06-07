"""One-shot builder for stage2_notebook_92_brief_bezier_ablation.ipynb.

Run this script ONCE to produce the notebook .ipynb file. Re-running it
overwrites the existing notebook. Keep this builder in version control
so the notebook can be regenerated cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [text],
    }


# ---------- Markdown intros ----------

INTRO = """\
# NB92 - Bezier + LCM brief ablation

3-row brief ablation comparing the polyline baseline against two Bezier
variants on a 10k stratified BDD subset (30 epochs each, batch=32). The
winner of this notebook becomes the headline config for NB90's full
250-epoch training on the full 70k+10k split.

| Row | Name | Representation | LCM | Hypothesis |
|---|---|---|---|---|
| 1 | `polyline_baseline_small` | 78-D polyline (= NB88 row 2 path)        | n/a | Apples-to-apples ref |
| 2 | `bezier_cubic_no_lcm`     | 16-D cubic Bezier                        | OFF | Does the representation help? |
| 3 | `bezier_lcm_gamma001`     | 16-D cubic Bezier + degree mixture       | ON, gamma=0.01 | Does adaptive degree help on top? |

The first three cells are self-contained: even on a FRESH Colab runtime
(useful because the user wants to run this in parallel with NB88 on a
DIFFERENT runtime), cells 1-3 mount Drive, extract a 10k subset, and
convert it to Bezier targets. After that, cells 5-7 launch the rows; you
can re-run any individual cell to retry a row.
"""

CELL1_MD = "### Cell 1: Mount Drive + locate sources"

CELL1_CODE = '''\
import os, sys, subprocess
from pathlib import Path
from google.colab import drive

os.environ['PYTHONIOENCODING'] = 'utf-8'

if not Path('/content/drive').exists():
    drive.mount('/content/drive', force_remount=False)

REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
if not os.path.isdir(REPO_ROOT):
    raise FileNotFoundError(f'Missing {REPO_ROOT} -- verify Drive sync.')
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import mmcv  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv'])
print('[ok] env ready')
'''

CELL2_MD = """\
### Cell 2: Extract a 10k stratified subset to /content/

Output layout (Colab\\'s local SSD, NOT Drive - the per-notebook-/content/
isolation rule means cells in DIFFERENT Colab runtimes don\\'t see each
other\\'s /content/):

```
/content/bdd_subset_10k/
    images/{train2017,val2017}/<stem>.jpg
    labels/{train2017,val2017}/<stem>.txt
    lane_targets/{train2017,val2017}/<stem>.pt    (78-D polyline)
```
"""

CELL2_CODE = '''\
import sys, subprocess
from pathlib import Path

PREP_SCRIPT = Path('stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py')
SUBSET_ROOT = Path('/content/bdd_subset_10k')

# Pre-flight: a missing prep script or missing Drive input is the most
# common cause of "exit status 2" with NO visible error - subprocess
# .check_call in Colab swallows the child's stderr when it dies on an
# argparse error or a "file not found". Surface those explicitly first.
if not PREP_SCRIPT.exists():
    raise FileNotFoundError(
        f'prep script not found relative to CWD={Path.cwd()}:\\n  {PREP_SCRIPT}\\n'
        'Verify the Drive copy of yolop_vehicle_lane is up to date.'
    )
required_inputs = [
    Path('/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'),
    Path('/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/BDD_detection_labels.zip'),
    Path('/content/drive/MyDrive/EcoCAR/datasets/lane_targets_clr_v1_polyline.tar.gz'),
]
missing = [str(p) for p in required_inputs if not p.exists()]
if missing:
    raise FileNotFoundError(
        'Required Drive input(s) missing - verify these exist on Drive:\\n  '
        + '\\n  '.join(missing)
    )

# Skip if already prepared (cheap restart-safety).
done_marker = SUBSET_ROOT / 'prep_summary.json'
if done_marker.exists():
    print(f'[ok] subset already prepared at {SUBSET_ROOT}; skipping')
else:
    cmd = [sys.executable, '-u', str(PREP_SCRIPT),
           '--out-root', str(SUBSET_ROOT),
           '--n-train', '10000', '--n-val', '2000', '--seed', '89']
    print('  cmd:', ' '.join(cmd), flush=True)
    # Stream stdout AND stderr (merged) line-by-line so the REAL error is
    # visible. Plain check_call hides the child's stderr in Colab when it
    # exits early (argparse / file-not-found), leaving only "exit 2".
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='', flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f'prepare_bdd_subset_10k.py exited with code {rc}. See the '
            f'streamed output above for the actual error. Common causes:\\n'
            f'  - a dataset/labels file moved or was renamed on Drive\\n'
            f'  - corrupted tar/zip download\\n'
            f'  - not enough free space on /content/ (~10 GB needed)'
        )

# Inventory
for label, d in (('images/train', SUBSET_ROOT / 'images/train2017'),
                 ('images/val',   SUBSET_ROOT / 'images/val2017'),
                 ('labels/train', SUBSET_ROOT / 'labels/train2017'),
                 ('labels/val',   SUBSET_ROOT / 'labels/val2017'),
                 ('lane_tgt/train', SUBSET_ROOT / 'lane_targets/train2017'),
                 ('lane_tgt/val',   SUBSET_ROOT / 'lane_targets/val2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {label:18s} {n}')
'''

CELL3_MD = """\
### Cell 3: Convert polyline (78-D) targets to Bezier (16-D) targets

Walks `lane_targets/{split}/<stem>.pt`, decodes each polyline back to
(x, y) coords, fits a cubic Bezier via least-squares, writes the 16-D
result to `lane_targets_bezier/{split}/<stem>.pt`. Reports a fit-error
histogram so we can sanity-check the representation before training
(target: median < 3 px, p95 < 8 px on a 640-px frame).
"""

CELL3_CODE = '''\
import sys, subprocess
from pathlib import Path

CONV_SCRIPT = 'stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/convert_polyline_pt_to_bezier_pt.py'
SUBSET_ROOT = Path('/content/bdd_subset_10k')

bezier_dir = SUBSET_ROOT / 'lane_targets_bezier'
fit_json = bezier_dir / 'fit_errors.json'
if fit_json.exists():
    print(f'[ok] Bezier targets already converted at {bezier_dir}; skipping')
else:
    cmd = [sys.executable, '-u', CONV_SCRIPT,
           '--src-root', str(SUBSET_ROOT / 'lane_targets'),
           '--dst-root', str(bezier_dir),
           '--img-size', '640',
           '--report-json', str(fit_json)]
    print('  cmd:', ' '.join(cmd))
    subprocess.check_call(cmd)

import json
report = json.loads(fit_json.read_text())
print('\\n[Bezier fit error report]')
for k, v in report.items():
    print(f'  {k:14s} {v}')

# Acceptance gate (per PLAN.md sec 3.5): median < 3 px (strict), p95 < 8 px (warning if violated)
if report['median_px'] > 3.0:
    print(f'\\n[WARN] median fit error {report["median_px"]:.2f} > 3 px - lanes may not Bezier-fit well')
if report['p95_px'] > 8.0:
    print(f'[WARN] p95 fit error {report["p95_px"]:.2f} > 8 px - some lanes have noisy fits')
'''

CELL4_MD = """\
### Cell 4: Helper - `launch_brief(name)` for any of the 3 rows

Self-contained: re-establishes the globals after a kernel restart.
"""

CELL4_CODE = '''\
"""Self-contained helper that re-establishes all globals after a kernel
restart. Defines `launch_brief(name, fresh=False)` for the 3 ablation rows.
"""
import os, sys, subprocess, shutil
from pathlib import Path

REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
if not os.path.isdir(REPO_ROOT):
    raise FileNotFoundError(f'Missing {REPO_ROOT} -- verify Drive sync.')
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import mmcv  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv'])

from stage2.scripts.notebook_utils import run_streaming

LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)
DRIVE_RUNS = Path('/content/drive/MyDrive/EcoCAR/training_runs/nb92_brief')
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

VENDOR_MOD = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr'
VENDOR_DATA = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/datasets'

ROW_CONFIGS = {
    'polyline_baseline_small': {
        'model': VENDOR_MOD / 'rtdetr-l_bdd_clr_lane.yaml',
        'data':  VENDOR_DATA / 'BDD_lane_only_10k.yaml',
    },
    'bezier_cubic_no_lcm': {
        'model': VENDOR_MOD / 'rtdetr-l_bdd_bezier_cubic.yaml',
        'data':  VENDOR_DATA / 'BDD_lane_only_bezier.yaml',
    },
    'bezier_lcm_gamma001': {
        'model': VENDOR_MOD / 'rtdetr-l_bdd_bezier_lcm.yaml',
        'data':  VENDOR_DATA / 'BDD_lane_only_bezier.yaml',
    },
}

# Confirm cells 2-3 are done so launchers can be run in any order.
SUBSET_ROOT = Path('/content/bdd_subset_10k')
if not (SUBSET_ROOT / 'images/train2017').exists():
    print('[warn] /content/bdd_subset_10k/ missing - re-run cell 2 first.')
elif not (SUBSET_ROOT / 'lane_targets_bezier/fit_errors.json').exists():
    print('[warn] Bezier targets missing - re-run cell 3 first.')
else:
    n_train = sum(1 for _ in (SUBSET_ROOT / 'images/train2017').iterdir())
    n_val   = sum(1 for _ in (SUBSET_ROOT / 'images/val2017').iterdir())
    print(f'[ok] subset on /content/: {n_train} train, {n_val} val')

# Defensive YAML rewrite: enforce both `path:` and `lane_targets_root:`
# at runtime so neither row can silently train on the wrong directory.
# This guards against the failure mode hit on the first NB92 run, where
# the YAMLs had stale lane_targets_root values, the dataset fell back to
# PNG-mask mode, cv2.imread returned None, and training died with
# "TypeError: '>' not supported between instances of 'NoneType' and 'int'".
def _rewrite_yaml_field(yaml_path: Path, field: str, expected_value: str):
    """Replace the first line starting with `<field>:` with the expected value."""
    txt = yaml_path.read_text(encoding='utf-8')
    target_line = f'{field}: {expected_value}'
    if target_line in txt:
        return False  # already correct
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f'{field}:'):
            lines[i] = target_line
            yaml_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
            print(f'  [yaml] rewrote {yaml_path.name}: {field} -> {expected_value}')
            return True
    # field not present - append
    lines.append(target_line)
    yaml_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
    print(f'  [yaml] added {yaml_path.name}: {field} -> {expected_value}')
    return True

# Polyline row uses /content/bdd_subset_10k/lane_targets/ (78-D .pt files
# produced by prepare_bdd_subset.py).
_rewrite_yaml_field(ROW_CONFIGS['polyline_baseline_small']['data'],
                    'path', '/content/bdd_subset_10k')
_rewrite_yaml_field(ROW_CONFIGS['polyline_baseline_small']['data'],
                    'lane_targets_root', '/content/bdd_subset_10k/lane_targets')

# Bezier rows use /content/bdd_subset_10k/lane_targets_bezier/ (16-D .pt
# files produced by convert_polyline_pt_to_bezier_pt.py in cell 3).
for _r in ('bezier_cubic_no_lcm', 'bezier_lcm_gamma001'):
    _rewrite_yaml_field(ROW_CONFIGS[_r]['data'], 'path', '/content/bdd_subset_10k')
    _rewrite_yaml_field(ROW_CONFIGS[_r]['data'], 'lane_targets_root',
                        '/content/bdd_subset_10k/lane_targets_bezier')

# Pre-flight: confirm the lane targets actually exist on disk for each
# row. If they don't, the dataset will silently fall back to PNG masks
# (which also don't exist on the subset), producing the cryptic
# "'>' not supported between NoneType and int" error during training.
def _check_row_targets(name: str):
    data_yaml = ROW_CONFIGS[name]['data']
    txt = data_yaml.read_text(encoding='utf-8')
    lt_root = None
    for ln in txt.splitlines():
        if ln.strip().startswith('lane_targets_root:'):
            lt_root = ln.split(':', 1)[1].strip()
            break
    if not lt_root:
        print(f'  [warn] {name}: no lane_targets_root in {data_yaml.name}')
        return
    for split in ('train2017', 'val2017'):
        d = Path(lt_root) / split
        n = sum(1 for _ in d.iterdir()) if d.exists() else 0
        marker = '[ok] ' if n > 0 else '[ERR]'
        print(f'  {marker} {name}: {lt_root}/{split} -> {n} .pt files')
        if n == 0:
            raise FileNotFoundError(
                f'No lane targets found at {d}. For polyline rows re-run cell 2; '
                f'for bezier rows re-run cell 3.'
            )

for _r in ROW_CONFIGS:
    _check_row_targets(_r)

TRAIN_SCRIPT = 'stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/train_bezier_brief.py'


def launch_brief(name: str, batch: int = 32, epochs: int = 30, lr0: float = 4e-4,
                 workers: int = 8, fresh: bool = False) -> bool:
    """Run one brief-ablation row.

    Resume-safe: re-running picks up from the last checkpoint. Pass
    fresh=True to wipe the run dir first (use this if the previous run
    produced poisoned weights and you want a clean start).
    """
    cfg = ROW_CONFIGS[name]
    for k, p in cfg.items():
        if not p.exists():
            raise FileNotFoundError(f'Missing {k} for {name}: {p}')
    run_dir_local = Path('/content/runs') / name
    if fresh and run_dir_local.exists():
        print(f'[fresh] wiping {run_dir_local}', flush=True)
        shutil.rmtree(run_dir_local, ignore_errors=True)
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--name', name,
        '--project', '/content/runs',
        '--model-yaml', str(cfg['model']),
        '--data-yaml',  str(cfg['data']),
        '--device', '0',
        '--workers', str(workers),
        '--save-period', '5',
        '--batch', str(batch),
        '--epochs', str(epochs),
        '--lr0', str(lr0),
        '--lcm-warning-on-row-mismatch',
    ]
    log = os.path.join(LOG_DIR, f'NB92_{name}.log')
    print(f'\\n=== launching {name} (epochs={epochs}, batch={batch}, lr0={lr0}, fresh={fresh}) ===\\n', flush=True)
    rc = run_streaming(cmd, log_path=log, check=False)
    if rc != 0:
        print(f'[warn] {name} training rc={rc}; see {log}.')
        return False
    if run_dir_local.exists():
        drive_tar = DRIVE_RUNS / f'{name}.tar.gz'
        print(f'[sync] tar {run_dir_local} -> {drive_tar}', flush=True)
        subprocess.check_call(['tar', '-C', '/content/runs', '-czf', str(drive_tar), name])
        print(f'  size: {drive_tar.stat().st_size / 1e6:.1f} MB')
    return True


print('\\n[ready] call `launch_brief("<row name>", fresh=True)` to start. Row names:')
for k in ROW_CONFIGS:
    print(f'  - {k}')
'''

CELL5_MD = "### Cell 5: Row 1 - `polyline_baseline_small` (~25 min)"
CELL5_CODE = '''\
ok = launch_brief('polyline_baseline_small', batch=32, epochs=30, lr0=4e-4, fresh=True)
print('row 1 done' if ok else 'row 1 incomplete (resume by re-running with fresh=False)')
'''

CELL6_MD = "### Cell 6: Row 2 - `bezier_cubic_no_lcm` (~25 min)"
CELL6_CODE = '''\
ok = launch_brief('bezier_cubic_no_lcm', batch=32, epochs=30, lr0=4e-4, fresh=True)
print('row 2 done' if ok else 'row 2 incomplete (resume by re-running with fresh=False)')
'''

CELL7_MD = "### Cell 7: Row 3 - `bezier_lcm_gamma001` (~25 min)"
CELL7_CODE = '''\
ok = launch_brief('bezier_lcm_gamma001', batch=32, epochs=30, lr0=4e-4, fresh=True)
print('row 3 done' if ok else 'row 3 incomplete (resume by re-running with fresh=False)')
'''

CELL8_MD = """\
### Cell 8: Aggregate the 3 rows and pick the winner

Reads each row's `runs/<name>/results.csv`, extracts the headline
metrics (mAP50 for det, lane IoU / ACC / F1 for the lane branch), and
writes `ablation_brief_results.md`. Announces the winner for NB90.
"""

CELL8_CODE = '''\
import csv, json
from pathlib import Path

def _load_final_row(run_dir: Path):
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None

def _g(row, *keys):
    if row is None:
        return None
    for k in keys:
        for ck in row:
            if ck.strip().lower() == k.lower():
                try:
                    return float(row[ck])
                except (TypeError, ValueError):
                    return row[ck]
    return None

OUT = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/extensions/bezier_lcm/ablation_brief_results.md'
rows_data = {}
for name in ROW_CONFIGS:
    rows_data[name] = _load_final_row(Path('/content/runs') / name)

# Pull metrics, pick winner by lane IoU (fall back to F1 then mAP50).
records = []
for name in ROW_CONFIGS:
    r = rows_data[name]
    map50 = _g(r, 'metrics/mAP50', 'mAP50', 'mAP_0.5')
    iou   = _g(r, 'metrics/IoU(lane)', 'iou', 'metrics/iou')
    acc   = _g(r, 'metrics/ACC(lane)', 'acc')
    f1    = _g(r, 'metrics/F1(lane)', 'f1')
    records.append({'name': name, 'mAP50': map50, 'IoU': iou, 'ACC': acc, 'F1': f1})

def _score(rec):
    # Primary key: lane IoU. Tiebreaker: F1 then ACC then mAP50.
    return (rec['IoU'] or -1, rec['F1'] or -1, rec['ACC'] or -1, rec['mAP50'] or -1)

records.sort(key=_score, reverse=True)
winner = records[0] if records[0]['IoU'] is not None else None

lines = ['# NB92 brief ablation - results\\n']
lines.append('Budget: 30 epochs, batch=32, lr0=4e-4, 10k train + 2k val.\\n')
lines.append('| Config | mAP50 | IoU (lane) | ACC (lane) | F1 (lane) |')
lines.append('|--------|-------|------------|------------|-----------|')
for rec in records:
    def _fmt(x):
        return f'{x:.4f}' if isinstance(x, float) else (x or 'TBD')
    lines.append(f"| `{rec['name']}` | {_fmt(rec['mAP50'])} | {_fmt(rec['IoU'])} | {_fmt(rec['ACC'])} | {_fmt(rec['F1'])} |")

if winner:
    lines.append(f'\\n## Winner: `{winner["name"]}`')
    lines.append(f'\\n- IoU = {winner["IoU"]}')
    lines.append('\\nForward this config to NB90 for the full 250-epoch run on the 70k train split.')
else:
    lines.append('\\n## No clear winner yet - all rows still TBD. Verify training completed.')

OUT.write_text('\\n'.join(lines), encoding='utf-8')
print(f'[wrote] {OUT}')
print('\\n' + '\\n'.join(lines))
'''


def build():
    nb = {
        "cells": [
            md(INTRO),
            md(CELL1_MD), code(CELL1_CODE),
            md(CELL2_MD), code(CELL2_CODE),
            md(CELL3_MD), code(CELL3_CODE),
            md(CELL4_MD), code(CELL4_CODE),
            md(CELL5_MD), code(CELL5_CODE),
            md(CELL6_MD), code(CELL6_CODE),
            md(CELL7_MD), code(CELL7_CODE),
            md(CELL8_MD), code(CELL8_CODE),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_92_brief_bezier_ablation.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
