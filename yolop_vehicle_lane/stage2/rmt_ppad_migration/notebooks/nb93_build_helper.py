"""Builder for stage2_notebook_93_bezier_full_training.ipynb.

NB93 takes the WINNER from NB92's brief ablation and trains it for the
full 250 epochs on the full 70k+10k BDD split.

The actual winner config (polyline / bezier-cubic / bezier-lcm) is
selected by the `WINNER` variable in cell 3 - just edit that string
after looking at NB92's `ablation_brief_results.md`.
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


INTRO = """\
# NB93 - Full training of the Bezier+LCM winner from NB92

After NB92's brief ablation picks a winner (polyline / bezier_cubic /
bezier_lcm), this notebook trains THAT config for the full 250 epochs
on the full 70k train + 10k val BDD split, with the same per-epoch
Drive sync + metrics-table mechanism as NB88/89/90.

Set `WINNER` in cell 3 to the row name announced by NB92.

Total wall time projection at batch=32: ~30 hours on an RTX Pro 6000.
"""


def build():
    nb = {
        "cells": [
            md(INTRO),

            md("### Cell 1: Mount Drive + locate sources"),
            code("""\
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
"""),

            md("""\
### Cell 2: Extract the FULL BDD dataset to /content/

Reuses the same `prepare_bdd_subset.py` script as NB88/89/90 with
`n_train=70000, n_val=10000`. If the Bezier winner needs Bezier targets,
this cell will also run the polyline-to-Bezier converter on the full
set.
"""),
            code("""\
import sys, subprocess
from pathlib import Path

PREP_SCRIPT = 'stage2/rmt_ppad_migration/P8_train/scripts/prepare_bdd_subset.py'
SUBSET_ROOT = Path('/content/bdd_dataset')

if not (SUBSET_ROOT / '.prep_done').exists():
    cmd = [sys.executable, '-u', PREP_SCRIPT,
           '--out-root', str(SUBSET_ROOT),
           '--n-train', '70000', '--n-val', '10000', '--seed', '93']
    print('  cmd:', ' '.join(cmd))
    subprocess.check_call(cmd)
    (SUBSET_ROOT / '.prep_done').write_text('ok')
else:
    print(f'[ok] subset already prepared at {SUBSET_ROOT}')

# Inventory
for label, d in (('images/train', SUBSET_ROOT / 'images/train2017'),
                 ('images/val',   SUBSET_ROOT / 'images/val2017'),
                 ('labels/train', SUBSET_ROOT / 'labels/train2017'),
                 ('labels/val',   SUBSET_ROOT / 'labels/val2017'),
                 ('lane_tgt/train', SUBSET_ROOT / 'lane_targets/train2017'),
                 ('lane_tgt/val',   SUBSET_ROOT / 'lane_targets/val2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {label:18s} {n}')
"""),

            md("""\
### Cell 3: Set the winner + (if Bezier) convert targets

Edit `WINNER` to match NB92's announcement. If the winner is a Bezier
variant, cell 4 needs `lane_targets_bezier/` on /content/; this cell
runs the polyline-to-Bezier converter on the full set.
"""),
            code("""\
# === EDIT ME based on NB92's ablation_brief_results.md ===
# Options: 'polyline_full', 'bezier_cubic_full', 'bezier_lcm_full'
WINNER = 'bezier_lcm_full'

assert WINNER in ('polyline_full', 'bezier_cubic_full', 'bezier_lcm_full')

NEEDS_BEZIER = WINNER in ('bezier_cubic_full', 'bezier_lcm_full')
print(f'[winner] {WINNER}  needs_bezier={NEEDS_BEZIER}')

if NEEDS_BEZIER:
    import subprocess
    CONV_SCRIPT = 'stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/convert_polyline_pt_to_bezier_pt.py'
    SUBSET_ROOT = Path('/content/bdd_dataset')
    fit_json = SUBSET_ROOT / 'lane_targets_bezier' / 'fit_errors.json'
    if fit_json.exists():
        print(f'[ok] Bezier targets already converted')
    else:
        cmd = [sys.executable, '-u', CONV_SCRIPT,
               '--src-root', str(SUBSET_ROOT / 'lane_targets'),
               '--dst-root', str(SUBSET_ROOT / 'lane_targets_bezier'),
               '--img-size', '640',
               '--report-json', str(fit_json)]
        print('  cmd:', ' '.join(cmd))
        subprocess.check_call(cmd)
"""),

            md("""\
### Cell 4: Launch the full 250-epoch training of the winner

Uses the same per-epoch Drive sync + metrics-table callbacks as
NB88-90. Checkpoints land at
`/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/<WINNER>/`
in real time.
"""),
            code("""\
import os, sys, shutil, subprocess
from pathlib import Path

# Map WINNER -> (model_yaml, data_yaml, train_script).
VENDOR_MOD  = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr'
VENDOR_DATA = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/datasets'

WINNER_TO_CFG = {
    'polyline_full': {
        'model': VENDOR_MOD / 'rtdetr-l_bdd_clr_lane.yaml',
        'data':  VENDOR_DATA / 'BDD_lane_only.yaml',
        'script': 'stage2/rmt_ppad_migration/P8_train/scripts/train_lane_only.py',
        'extra_args': ['--mode', 'full'],
    },
    'bezier_cubic_full': {
        'model': VENDOR_MOD / 'rtdetr-l_bdd_bezier_cubic.yaml',
        'data':  VENDOR_DATA / 'BDD_lane_only_bezier.yaml',
        'script': 'stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/train_bezier_brief.py',
        'extra_args': [],
    },
    'bezier_lcm_full': {
        'model': VENDOR_MOD / 'rtdetr-l_bdd_bezier_lcm.yaml',
        'data':  VENDOR_DATA / 'BDD_lane_only_bezier.yaml',
        'script': 'stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/train_bezier_brief.py',
        'extra_args': [],
    },
}

cfg = WINNER_TO_CFG[WINNER]

# Rewrite the Bezier data YAML to point at /content/bdd_dataset rather
# than /content/bdd_subset_bezier (which NB92 used).
def _rewrite_yaml_path(yaml_path, expected_path):
    if not yaml_path.exists():
        return
    txt = yaml_path.read_text()
    out_lines = []
    for ln in txt.splitlines():
        if ln.startswith('path:'):
            out_lines.append(f'path: {expected_path}')
        elif 'lane_targets_root:' in ln and 'bezier' in str(yaml_path):
            out_lines.append('lane_targets_root: /content/bdd_dataset/lane_targets_bezier')
        else:
            out_lines.append(ln)
    yaml_path.write_text('\\n'.join(out_lines) + '\\n')

_rewrite_yaml_path(cfg['data'], '/content/bdd_dataset')

# Fresh-start switch. Default True; flip to False to resume from a
# Drive-saved checkpoint (then run the optional restore cell below).
FRESH = True
run_dir_local = Path('/content/runs') / WINNER
if FRESH and run_dir_local.exists():
    shutil.rmtree(run_dir_local, ignore_errors=True)

LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)

from stage2.scripts.notebook_utils import run_streaming

cmd = [
    sys.executable, '-u', cfg['script'],
    *cfg['extra_args'],
    '--name', WINNER,
    '--project', '/content/runs',
    '--model-yaml', str(cfg['model']),
    '--data-yaml', str(cfg['data']),
    '--device', '0',
    '--workers', '8',
    '--save-period', '10',
    '--batch', '32',
    '--epochs', '250',
    '--lr0', '4e-4',
]
log = os.path.join(LOG_DIR, f'NB93_{WINNER}.log')
print(f'=== launching {WINNER} (full 250 epochs, batch=32, lr0=4e-4) ===\\n')
rc = run_streaming(cmd, log_path=log, check=False)
print(f'training rc={rc}; log -> {log}')
"""),

            md("""\
### Cell 5 (optional): Resume from Drive after disconnect

Same restore pattern as NB88-90. After this cell completes, re-run
cell 4 with `FRESH = False`.
"""),
            code("""\
from pathlib import Path
import shutil

drive_ckpt = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints') / WINNER
local_weights = Path('/content/runs') / WINNER / 'weights'
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
print(f'[restore] {restored or "(nothing to restore)"}')
"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_93_bezier_full_training.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
