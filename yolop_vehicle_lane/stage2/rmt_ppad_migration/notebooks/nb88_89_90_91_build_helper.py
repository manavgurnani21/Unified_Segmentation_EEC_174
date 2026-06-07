"""Build the 4 P8-ablation notebooks: NB88, NB89, NB90 (one row each)
and NB91 (aggregator).

Each of NB88/89/90 trains a SINGLE ablation row on the FULL 70k train +
10k val BDD subset at batch=32, lr0=4e-4. Designed to run on three
SEPARATE Colab runtimes in parallel - each notebook is self-contained
(mounts Drive, extracts the dataset itself, etc.).

NB91 mounts Drive and reads the results from
`/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/<row>/results.csv`
so the table can be built no matter which runtime trained which row.

Re-run this builder any time you want to regenerate the .ipynb files.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [text],
    }


# ---------------------------------------------------------------------------
# Shared cell sources
# ---------------------------------------------------------------------------

ROW_DESCRIPTIONS = {
    'clr_lane_default': """\
**Row 2** of the P8 ablation. Default config: GCA on,
`CLRHeadForSquareImage` (32/128/32 priors), `max_lanes=8`. Headline
question: does the full pipeline beat the polyline-mask baseline?""",
    'clr_lane_no_square_priors': """\
**Row 3** of the P8 ablation. Base `CLRHead` with CULane's 24/144/24
prior split (instead of the square-image-tuned 32/128/32). Headline
question: does the prior retuning matter, or is the CULane default fine?""",
    'clr_lane_no_gca': """\
**Row 4** of the P8 ablation. `LaneSegHead` consumes raw `input_proj`
features (`task_adapter_seg` + `gate_seg` bypassed for the lane path).
Headline question: does GCA help the lane branch, or are raw FPN
features sufficient?""",
}

ROW_YAMLS = {
    'clr_lane_default': 'rtdetr-l_bdd_clr_lane.yaml',
    'clr_lane_no_square_priors': 'rtdetr-l_bdd_clr_lane_no_square_priors.yaml',
    'clr_lane_no_gca': 'rtdetr-l_bdd_clr_lane_no_gca.yaml',
}

NB_IDS = {
    'clr_lane_default': 88,
    'clr_lane_no_square_priors': 89,
    'clr_lane_no_gca': 90,
}


def make_row_notebook(row_name: str) -> dict:
    nb_id = NB_IDS[row_name]
    yaml_file = ROW_YAMLS[row_name]
    description = ROW_DESCRIPTIONS[row_name]

    intro = f"""\
# NB{nb_id} - P8 ablation row: `{row_name}`

{description}

**Self-contained**: this notebook mounts Drive, extracts the full BDD
dataset to `/content/`, and launches a SINGLE training run. The other
two P8 ablation rows are in NB{{88, 89, 90}} \\ this one - run them on
separate Colab runtimes for parallelism. NB91 aggregates the 3 results
into the final ablation table.

**Per-epoch saves**: `last.pt`, `best.pt` (only when val improves), and
a `log_epoch_NNN.json` metrics snapshot are mirrored to
`/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/{row_name}/`
at the end of every epoch. Safe against Colab disconnects.

**Per-epoch metrics table**: validation runs after every epoch and a
fixed-width row of detection + lane metrics is printed below the loss
columns.
"""

    cell1_md = "### Cell 1: Mount Drive + locate sources"
    cell1_code = """\
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
"""

    cell2_md = """\
### Cell 2: Extract FULL BDD dataset to /content/

70k train + 10k val images + matching detection labels + matching
(max_lanes, 78) polyline targets. Same prep script the smoke / brief
notebooks use, just with `n_train=70000, n_val=10000`.

The dataset goes on `/content/` (Colab's local SSD) - NEVER on Drive,
per the project's Drive-vs-local rule. Each Colab runtime extracts its
own copy.
"""
    cell2_code = """\
import sys, subprocess
from pathlib import Path

PREP_SCRIPT = Path('stage2/rmt_ppad_migration/P8_train/scripts/prepare_bdd_subset.py')
SUBSET_ROOT = Path('/content/bdd_dataset')

# Pre-flight: missing prep script or missing Drive inputs are the most
# common causes of "exit 2" with no visible stderr (subprocess.check_call
# in Colab can swallow the child's stderr on argparse / "file not found"
# style failures). Surface them explicitly before launching.
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
        'Required Drive input(s) missing - verify these exist:\\n  '
        + '\\n  '.join(missing)
    )

if (SUBSET_ROOT / '.prep_done').exists():
    print(f'[ok] subset already prepared at {SUBSET_ROOT}; skipping')
else:
    cmd = [sys.executable, '-u', str(PREP_SCRIPT),
           '--out-root', str(SUBSET_ROOT),
           '--n-train', '70000', '--n-val', '10000', '--seed', '88']
    print('  cmd:', ' '.join(cmd), flush=True)
    # Stream stdout AND stderr (merged) line-by-line. subprocess.check_call
    # alone hides stderr in Colab when the child exits early - if the
    # child failed with "file not found" or argparse error, that error
    # is lost without this merging.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='', flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f'prepare_bdd_subset.py exited with code {rc}. See the streamed '
            f'output above for the actual error. Common causes:\\n'
            f'  - dataset / labels file moved on Drive\\n'
            f'  - corrupted tar/zip download\\n'
            f'  - not enough free space on /content/ (~30 GB needed)'
        )
    (SUBSET_ROOT / '.prep_done').write_text('ok')

for label, d in (('images/train', SUBSET_ROOT / 'images/train2017'),
                 ('images/val',   SUBSET_ROOT / 'images/val2017'),
                 ('labels/train', SUBSET_ROOT / 'labels/train2017'),
                 ('labels/val',   SUBSET_ROOT / 'labels/val2017'),
                 ('lane_tgt/train', SUBSET_ROOT / 'lane_targets/train2017'),
                 ('lane_tgt/val',   SUBSET_ROOT / 'lane_targets/val2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {label:18s} {n}')
"""

    cell3_md = f"""\
### Cell 3: Launch the `{row_name}` training run

batch=32, lr0=4e-4, AMP off, spatial-aug off, photometric-aug on. The
training script registers the per-epoch metrics table + Drive sync
automatically; this cell just kicks it off.
"""

    cell3_code = f"""\
import os, sys, subprocess, shutil
from pathlib import Path

ROW_NAME = '{row_name}'
YAML = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/{yaml_file}'
TRAIN_SCRIPT = 'stage2/rmt_ppad_migration/P8_train/scripts/train_lane_only.py'
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Optional fresh-start switch - set to True to wipe /content/runs/<row>/
# and start from epoch 0. Default is resume-from-Drive (see cell 4).
FRESH = True

run_dir_local = Path('/content/runs') / ROW_NAME
if FRESH and run_dir_local.exists():
    print(f'[fresh] wiping {{run_dir_local}}')
    shutil.rmtree(run_dir_local, ignore_errors=True)

if not YAML.exists():
    raise FileNotFoundError(f'Missing YAML: {{YAML}}')

from stage2.scripts.notebook_utils import run_streaming

cmd = [
    sys.executable, '-u', TRAIN_SCRIPT,
    '--mode', 'full',
    '--name', ROW_NAME,
    '--project', '/content/runs',
    '--model-yaml', str(YAML),
    '--device', '0',
    '--workers', '8',
    '--save-period', '10',
    '--batch', '32',
    '--epochs', '250',
    '--lr0', '4e-4',
]
log = os.path.join(LOG_DIR, f'NB{nb_id}_{{ROW_NAME}}.log')
print(f'=== launching {{ROW_NAME}} (batch=32, lr0=4e-4, epochs=250) ===\\n', flush=True)
rc = run_streaming(cmd, log_path=log, check=False)
print(f'training rc={{rc}}; log -> {{log}}')
"""

    cell4_md = """\
### Cell 4 (optional): Resume from a previous Drive-saved checkpoint

If Colab disconnected mid-training, the per-epoch Drive sync left
`last.pt` and `best.pt` at
`/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/<row>/`. This
cell pulls that snapshot back into `/content/runs/<row>/weights/` so
re-running cell 3 with `FRESH = False` will pick up at the saved
epoch.
"""

    cell4_code = """\
import shutil
from pathlib import Path

drive_ckpt = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints') / ROW_NAME
local_weights = Path('/content/runs') / ROW_NAME / 'weights'
local_weights.mkdir(parents=True, exist_ok=True)

restored = []
for pt_name in ('last.pt', 'best.pt'):
    src = drive_ckpt / pt_name
    if src.exists():
        shutil.copy2(src, local_weights / pt_name)
        restored.append(pt_name)
results_csv = drive_ckpt / 'results.csv'
if results_csv.exists():
    shutil.copy2(results_csv, local_weights.parent / 'results.csv')
    restored.append('results.csv')

print(f'[restore] from {drive_ckpt}:')
for name in restored:
    p = local_weights / name if name.endswith('.pt') else local_weights.parent / name
    print(f'  {name}  ({p.stat().st_size / 1e6:.1f} MB)')
if not restored:
    print('  (nothing to restore - drive dir empty)')
print('\\nNow re-run cell 3 with FRESH = False to resume training.')
"""

    return {
        "cells": [
            md(intro),
            md(cell1_md), code(cell1_code),
            md(cell2_md), code(cell2_code),
            md(cell3_md), code(cell3_code),
            md(cell4_md), code(cell4_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# NB91 aggregator
# ---------------------------------------------------------------------------

def make_nb91_aggregator() -> dict:
    intro = """\
# NB91 - P8 ablation aggregator

Reads the per-epoch `results.csv` for each of the 3 ablation rows from
`/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/<row>/` and
builds the final ablation table.

Run this notebook AFTER NB88, NB89, NB90 have produced (at least one
epoch of) results. You can run it at any point during training to
preview the current standings - just re-run the cell.
"""

    cell1_md = "### Cell 1: Mount Drive"
    cell1_code = """\
import os, sys
from pathlib import Path
from google.colab import drive

if not Path('/content/drive').exists():
    drive.mount('/content/drive', force_remount=False)

REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
os.chdir(REPO_ROOT)
print('[ok] mounted')
"""

    cell2_md = """\
### Cell 2: Read per-row results.csv + print the table

Picks the FINAL (last) row of each results.csv. If a run is still in
progress, this shows the latest epoch's metrics. Writes
`P8_train/results.md` so the table is git-trackable.
"""

    cell2_code = """\
import csv
from pathlib import Path

ROWS = ['clr_lane_default', 'clr_lane_no_square_priors', 'clr_lane_no_gca']
CKPT_ROOT = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints')

def _final_row(csv_path):
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None

def _g(row, *candidates):
    if row is None:
        return None
    for key_pat in candidates:
        for ck in row:
            if ck.strip().lower() == key_pat.lower():
                try:
                    return float(row[ck])
                except (TypeError, ValueError):
                    return row[ck]
    return None

records = []
for name in ROWS:
    r = _final_row(CKPT_ROOT / name / 'results.csv')
    records.append({
        'name': name,
        'epoch': _g(r, 'epoch'),
        'mAP50':    _g(r, 'metrics/mAP50(B)', 'metrics/mAP50', 'mAP50'),
        'mAP50-95': _g(r, 'metrics/mAP50-95(B)', 'metrics/mAP50-95'),
        'P(det)':   _g(r, 'metrics/precision(B)', 'metrics/precision'),
        'R(det)':   _g(r, 'metrics/recall(B)', 'metrics/recall'),
        'IoU(lane)': _g(r, 'metrics/IoU(lane)'),
        'mIoU(lane)': _g(r, 'metrics/mIoU(lane)'),
        'pixacc(lane)': _g(r, 'metrics/pixacc(lane)'),
        'subacc(lane)': _g(r, 'metrics/subacc(lane)'),
    })

lines = ['# P8 ablation table\\n']
lines.append('Pulled from `/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/<row>/results.csv` '
             '(end-of-last-completed-epoch).\\n')
lines.append('| Row | Epoch | mAP50 | mAP50-95 | P(det) | R(det) | IoU(lane) | mIoU | pixAcc | subAcc |')
lines.append('|-----|-------|-------|----------|--------|--------|-----------|------|--------|--------|')
def _fmt(x):
    if x is None: return 'TBD'
    if isinstance(x, float): return f'{x:.4f}'
    return str(x)
for rec in records:
    cells = [rec['name']] + [_fmt(rec[k]) for k in ('epoch', 'mAP50', 'mAP50-95',
                                                   'P(det)', 'R(det)',
                                                   'IoU(lane)', 'mIoU(lane)',
                                                   'pixacc(lane)', 'subacc(lane)')]
    lines.append('| ' + ' | '.join(f'`{cells[0]}`' if i == 0 else c for i, c in enumerate(cells)) + ' |')

print('\\n'.join(lines))

OUT = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/P8_train/results.md'
OUT.write_text('\\n'.join(lines), encoding='utf-8')
print(f'\\n[wrote] {OUT}')
"""

    return {
        "cells": [
            md(intro),
            md(cell1_md), code(cell1_code),
            md(cell2_md), code(cell2_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build():
    out_dir = Path(__file__).resolve().parent
    targets = [
        ('clr_lane_default', 'stage2_notebook_88_P8_clr_lane_default.ipynb'),
        ('clr_lane_no_square_priors', 'stage2_notebook_89_P8_clr_lane_no_square_priors.ipynb'),
        ('clr_lane_no_gca', 'stage2_notebook_90_P8_clr_lane_no_gca.ipynb'),
    ]
    for row, fname in targets:
        nb = make_row_notebook(row)
        path = out_dir / fname
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
        print(f'wrote {path}')

    nb91 = make_nb91_aggregator()
    p91 = out_dir / 'stage2_notebook_91_P8_aggregator.ipynb'
    p91.write_text(json.dumps(nb91, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {p91}')


if __name__ == '__main__':
    build()
