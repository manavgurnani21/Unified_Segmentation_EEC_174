"""One-shot builder for stage2_notebook_96_geometry_lever_ablation.ipynb.

NB96 = find the MINIMAL fix that unfreezes the lane geometry at a small-
but-real scale (2k images, 3 epochs/row). Only run after NB95 shows the
head CAN learn on 32 images. Toggles the env-driven knobs added to
train_lane_only.py: --lane-weights, --diff-clamp, --lane-match.
"""
from __future__ import annotations
import json
from pathlib import Path


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": [t]}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [t]}


INTRO = """\
# NB96 - Geometry-lever ablation @ 2k images (diagnostic)

**Pipeline position:** after NB95 confirms the lane head CAN learn on a
tiny set, this finds the smallest config change that makes `IoU(lane)`
and `[lane-geom]` move at a representative 2k-image scale - the config to
carry into the post-fix full run (NB97+).

| row | lane-weights | diff-clamp | match | question |
|---|---|---|---|---|
| A | cut5x | 100  | hungarian | reproduce the NB94 frozen baseline (control) |
| B | clrkd | none | hungarian | does strong, UNCLAMPED geometry supervision move IoU? |
| C | clrkd | none | dynamic_k | does DENSER matching (more positive priors) help geometry? |

3 epochs/row, batch 32. Watch each row's `IoU(lane)` slope and whether
`[lane-geom]` means move. The winning row = the full-run config.
"""

CELL1 = '''\
import os, sys, subprocess
from pathlib import Path
from google.colab import drive
os.environ['PYTHONIOENCODING'] = 'utf-8'
if not Path('/content/drive').exists(): drive.mount('/content/drive', force_remount=False)
REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
if not os.path.isdir(REPO_ROOT): raise FileNotFoundError(f'Missing {REPO_ROOT}')
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)
try:
    import scipy, mmcv  # noqa
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv', 'scipy'])
print('[ok] env ready')
'''

CELL2_MD = "### Cell 2: Prep 2k-image subset (2000 train + 500 val)"
CELL2 = '''\
import sys, subprocess
from pathlib import Path
PREP = Path('stage2/rmt_ppad_migration/P8_train/scripts/prepare_bdd_subset.py')
ROOT = Path('/content/bdd_subset_2k')
if not PREP.exists(): raise FileNotFoundError(f'prep missing: {PREP} (sync Drive)')
if (ROOT/'prep_summary.json').exists():
    print(f'[ok] 2k subset present at {ROOT}; skipping')
else:
    cmd = [sys.executable, '-u', str(PREP), '--out-root', str(ROOT),
           '--n-train', '2000', '--n-val', '500', '--seed', '96']
    print('  cmd:', ' '.join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout: print(line, end='', flush=True)
    if proc.wait() != 0: raise RuntimeError('prep failed; see output above')
for lab, d in (('img/train', ROOT/'images/train2017'), ('img/val', ROOT/'images/val2017'),
               ('lt/train', ROOT/'lane_targets/train2017'), ('lt/val', ROOT/'lane_targets/val2017')):
    print(f'  {lab:10s}', sum(1 for _ in d.iterdir()) if d.exists() else 0)
'''

CELL3_MD = "### Cell 3: Helper - `run_row(name, weights, clamp, match)`"
CELL3 = '''\
import os, sys, subprocess, shutil
from pathlib import Path
from stage2.scripts.notebook_utils import run_streaming
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'; os.makedirs(LOG_DIR, exist_ok=True)
YAML_MODEL = Path(REPO_ROOT)/'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml'
YAML_DATA  = Path(REPO_ROOT)/'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_2k.yaml'
TRAIN = 'stage2/rmt_ppad_migration/P8_train/scripts/train_lane_only.py'

def run_row(name, weights, clamp, match, epochs=3, batch=32):
    rd = Path('/content/runs')/name
    if rd.exists(): shutil.rmtree(rd, ignore_errors=True)
    cmd = [sys.executable, '-u', TRAIN, '--mode', 'full', '--name', name,
           '--project', '/content/runs', '--model-yaml', str(YAML_MODEL),
           '--data-yaml', str(YAML_DATA), '--device', '0', '--workers', '8',
           '--save-period', '50', '--batch', str(batch), '--epochs', str(epochs),
           '--lr0', '4e-4', '--lane-match', match, '--lane-weights', weights,
           '--diff-clamp', clamp]
    log = os.path.join(LOG_DIR, f'NB96_{name}.log')
    print(f'\\n=== {name}: weights={weights} clamp={clamp} match={match} ===\\n', flush=True)
    rc = run_streaming(cmd, log_path=log, check=False)
    print(f'{name} rc={rc}')
    return rc
print('[ready] run_row defined')
'''

CELL4_MD = "### Cell 4: Row A - control (cut5x, clamp 100, hungarian) = NB94 baseline"
CELL4 = "run_row('abl_A_cut5x_clamp_hung', weights='cut5x', clamp='100', match='hungarian')\n"
CELL5_MD = "### Cell 5: Row B - clrkd weights, NO clamp, hungarian"
CELL5 = "run_row('abl_B_clrkd_noclamp_hung', weights='clrkd', clamp='none', match='hungarian')\n"
CELL6_MD = "### Cell 6: Row C - clrkd weights, no clamp, dynamic_k"
CELL6 = "run_row('abl_C_clrkd_noclamp_dynk', weights='clrkd', clamp='none', match='dynamic_k')\n"

CELL7_MD = """\
### Cell 7: Aggregate - which lever unfroze the geometry?

Reads each row's `full_train.log`, extracts the IoU(lane) trajectory and
the first/last `[lane-geom]` means, and prints a comparison + the winner.
"""
CELL7 = '''\
import re
from pathlib import Path
ROWS = ['abl_A_cut5x_clamp_hung', 'abl_B_clrkd_noclamp_hung', 'abl_C_clrkd_noclamp_dynk']
def parse(name):
    for base in (Path('/content/runs')/name/'full_train.log',
                 Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints')/name/'full_train.log'):
        if base.exists():
            txt = base.read_text(encoding='utf-8', errors='replace'); break
    else:
        return None
    ious = [float(m.group(6)) for l in txt.splitlines()
            if (m := re.match(r'\\s*(\\d+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)', l))]
    geoms = [tuple(float(x) for x in m.groups()) for l in txt.splitlines()
             if (m := re.search(r'start_x mean=([-\\d.]+).*theta mean=([-\\d.]+).*length mean=([-\\d.]+)', l))]
    iou_rng = (min(ious), max(ious)) if ious else (0, 0)
    geom_moved = len(geoms) >= 2 and any(abs(a-b) > 1e-4 for a, b in zip(geoms[0], geoms[-1]))
    return {'iou_min': iou_rng[0], 'iou_max': iou_rng[1], 'iou_moved': iou_rng[1]-iou_rng[0] > 0.01,
            'geom_moved': geom_moved, 'n_ep': len(ious)}

print(f'{"row":28s} {"IoU min..max":18s} {"IoU moved":10s} {"geom moved":10s}')
best = None
for r in ROWS:
    d = parse(r)
    if d is None: print(f'{r:28s} (no log - run its cell)'); continue
    print(f'{r:28s} {d["iou_min"]:.4f}..{d["iou_max"]:.4f}      {str(d["iou_moved"]):10s} {str(d["geom_moved"]):10s}')
    if d['iou_moved'] and (best is None): best = r
print('\\n================ VERDICT ================')
if best:
    print(f'  WINNER: {best} - carry its (weights, clamp, match) into the NB97+ full run.')
else:
    print('  No row moved IoU at 2k/3ep. If NB95 overfit DID move, try more epochs or')
    print('  a lane-only (freeze-detection) row; else the metric (Fault B) is masking it.')
'''


def build():
    nb = {"cells": [md(INTRO), md("### Cell 1: Mount + env"), code(CELL1),
                    md(CELL2_MD), code(CELL2), md(CELL3_MD), code(CELL3),
                    md(CELL4_MD), code(CELL4), md(CELL5_MD), code(CELL5),
                    md(CELL6_MD), code(CELL6), md(CELL7_MD), code(CELL7)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).resolve().parent / 'stage2_notebook_96_geometry_lever_ablation.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
