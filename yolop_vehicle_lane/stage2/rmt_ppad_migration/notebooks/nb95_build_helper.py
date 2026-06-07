"""One-shot builder for stage2_notebook_95_lane_overfit_probe.ipynb.

NB95 = the DECISIVE diagnostic: can the lane head learn lanes AT ALL?
Overfit a tiny fixed set (32 images, train==val) for many epochs with the
best-case geometry config (full CLRKDNet weights, no diff-clamp,
Hungarian). If IoU + [lane-geom] move -> architecture is sound, the
frozen-IoU-at-scale is a weight/optimization issue. If still frozen on 32
images -> structural bug; NB95 is then the minimal repro.

Run this script to (re)generate the .ipynb.
"""
from __future__ import annotations
import json
from pathlib import Path


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": [t]}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [t]}


INTRO = """\
# NB95 - Lane-head OVERFIT capability probe (diagnostic)

**Pipeline position:** full training (NB88-94) exposed a lane IoU frozen
at a per-val-set constant (0.0710/0.0718) even with Hungarian matching
active and the cls demonstrably learning. This notebook is the decisive
experiment BEFORE we spend any more GPU on full training: **can the lane
head learn lanes at all?**

**Method:** overfit 32 images (train == val), ~120 epochs, best-case
geometry config:
`--lane-weights clrkd` (full CLRKDNet weights, not the 5x-cut),
`--diff-clamp none` (the +-100 clamp may zero the geometry gradient),
`--lane-match hungarian`.

**Read the verdict from two signals (now logged every val epoch):**
- `IoU(lane)` in the metrics table - does it climb above the frozen ~0.07?
- `[lane-geom] start_x/theta/length mean+std` - do the geometry means
  MOVE across epochs (reg_layers learning) or stay byte-constant (frozen
  at the prior anchors)?

**Decision:**
- IoU climbs + geometry moves -> architecture SOUND; go to NB96 to find
  the minimal fix that survives at 2k images.
- IoU frozen on 32 images -> STRUCTURAL bug (train/eval forward mismatch,
  target-encoding error, gradient not reaching reg_layers). Single-step
  debug from here; do NOT scale up.
"""

CELL1 = '''\
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
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)
try:
    import scipy  # noqa: F401  (hungarian)
    import mmcv   # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv', 'scipy'])
print('[ok] env ready')
'''

CELL2_MD = """\
### Cell 2: Build the 32-image OVERFIT subset (train == val)

Prep 32 train images via the existing prep script, then COPY them into
`val2017` so train and val are the identical 32 images. A correct lane
head must memorize them and drive val IoU up.
"""

CELL2 = '''\
import sys, subprocess, shutil
from pathlib import Path

PREP = Path('stage2/rmt_ppad_migration/P8_train/scripts/prepare_bdd_subset.py')
ROOT = Path('/content/bdd_overfit32')
if not PREP.exists():
    raise FileNotFoundError(f'prep script missing: {PREP} (sync Drive)')

if (ROOT / 'prep_summary.json').exists():
    print(f'[ok] overfit subset already present at {ROOT}; skipping prep')
else:
    cmd = [sys.executable, '-u', str(PREP), '--out-root', str(ROOT),
           '--n-train', '32', '--n-val', '32', '--seed', '95']
    print('  cmd:', ' '.join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout: print(line, end='', flush=True)
    if proc.wait() != 0:
        raise RuntimeError('prep failed; see streamed output above')

# Force train == val: copy every train file into the matching val dir.
for sub in ('images/train2017', 'labels/train2017', 'lane_targets/train2017'):
    src = ROOT / sub
    dst = ROOT / sub.replace('train2017', 'val2017')
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name); n += 1
    print(f'  copied {n} files {sub} -> {dst}')
# Drop any stale val label cache so the new (train==val) files are rescanned.
for c in ROOT.rglob('*.cache'):
    c.unlink()
    print(f'  removed stale cache {c}')

for label, d in (('images/train', ROOT/'images/train2017'), ('images/val', ROOT/'images/val2017'),
                 ('lane_tgt/train', ROOT/'lane_targets/train2017'), ('lane_tgt/val', ROOT/'lane_targets/val2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {label:16s} {n}')
'''

CELL3_MD = """\
### Cell 3: Launch the overfit run (best-case geometry config)

120 epochs, batch 16, **full CLRKDNet lane weights**, **no diff-clamp**,
Hungarian. Per-epoch `[lane-geom]` + `IoU(lane)` go to the metrics table
and to `full_train.log` on Drive.
"""

CELL3 = '''\
import os, sys, subprocess
from pathlib import Path

ROW = 'probe_overfit32_clrkd_noclamp'
YAML_MODEL = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml'
YAML_DATA  = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_overfit.yaml'
TRAIN = 'stage2/rmt_ppad_migration/P8_train/scripts/train_lane_only.py'
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'; os.makedirs(LOG_DIR, exist_ok=True)
for p in (YAML_MODEL, YAML_DATA, Path(TRAIN)):
    if not Path(p).exists(): raise FileNotFoundError(f'missing {p} (sync Drive)')

import shutil
run_dir = Path('/content/runs') / ROW
if run_dir.exists(): shutil.rmtree(run_dir, ignore_errors=True)  # always fresh

from stage2.scripts.notebook_utils import run_streaming
cmd = [sys.executable, '-u', TRAIN, '--mode', 'full', '--name', ROW,
       '--project', '/content/runs', '--model-yaml', str(YAML_MODEL),
       '--data-yaml', str(YAML_DATA), '--device', '0', '--workers', '4',
       '--save-period', '40', '--batch', '16', '--epochs', '120', '--lr0', '4e-4',
       '--lane-match', 'hungarian', '--lane-weights', 'clrkd', '--diff-clamp', 'none']
log = os.path.join(LOG_DIR, f'NB95_{ROW}.log')
print('=== overfit probe: 32 img, 120 ep, clrkd weights, no clamp, hungarian ===\\n', flush=True)
rc = run_streaming(cmd, log_path=log, check=False)
print(f'rc={rc}; log -> {log}')
'''

CELL4_MD = """\
### Cell 4: Verdict - did the geometry and IoU move?

Parses `full_train.log` for the per-epoch `IoU(lane)` and `[lane-geom]`
lines and prints the trajectory + an automatic verdict.
"""

CELL4 = '''\
import re
from pathlib import Path

ROW = 'probe_overfit32_clrkd_noclamp'
log = Path('/content/runs') / ROW / 'full_train.log'
if not log.exists():
    log = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints') / ROW / 'full_train.log'
txt = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
if not txt:
    print('[warn] no full_train.log found; re-run cell 3'); raise SystemExit

ious, geoms = [], []
for l in txt.splitlines():
    m = re.search(r'start_x mean=([-\\d.]+).*theta mean=([-\\d.]+).*length mean=([-\\d.]+)', l)
    if m: geoms.append(tuple(float(x) for x in m.groups()))
# metrics-table IoU column (6th float after the epoch int)
for l in txt.splitlines():
    m = re.match(r'\\s*(\\d+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)', l)
    if m: ious.append((int(m.group(1)), float(m.group(6))))

print('epoch  IoU(lane)')
for ep, iou in ious[:60]: print(f'  {ep:3d}   {iou:.4f}')
print('\\nstart_x / theta / length means (first vs last):')
if geoms:
    print('  first:', geoms[0]); print('  last :', geoms[-1])

iou_vals = [v for _, v in ious]
iou_moved = (max(iou_vals) - min(iou_vals) > 0.01) if iou_vals else False
geom_moved = (len(geoms) >= 2 and any(abs(a-b) > 1e-4 for a, b in zip(geoms[0], geoms[-1])))
print('\\n================ VERDICT ================')
print(f'  IoU moved >0.01:   {iou_moved}  (range {min(iou_vals):.4f}..{max(iou_vals):.4f})' if iou_vals else '  no IoU rows')
print(f'  geometry moved:    {geom_moved}')
if iou_moved and geom_moved:
    print('  => ARCHITECTURE SOUND. Frozen-at-scale is a weight/optim issue. Proceed to NB96.')
elif geom_moved and not iou_moved:
    print('  => geometry moves but IoU frozen => METRIC fault (Fault B). Build curve-F1 metric.')
else:
    print('  => CANNOT overfit 32 imgs => STRUCTURAL bug. Debug train/eval forward + target encoding here.')
'''


def build():
    nb = {"cells": [md(INTRO), md("### Cell 1: Mount + env"), code(CELL1),
                    md(CELL2_MD), code(CELL2), md(CELL3_MD), code(CELL3),
                    md(CELL4_MD), code(CELL4)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).resolve().parent / 'stage2_notebook_95_lane_overfit_probe.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
