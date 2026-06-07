"""One-shot builder for stage2_notebook_100_curveiou_revalidation.ipynb.

NB100 - re-validate the 7 NB98 ablation checkpoints with the FIXED curveIoU
metric and print the corrected ranking. NO retraining: it loads each saved
best.pt and runs a validation pass only (~minutes/row).

WHY: NB98 reported curveIoU=0 for every row because of two metric bugs (now
fixed in P7_validator/tools/lane_curve_f1.py): (1) curveIoU was scored only
over the top-8-by-score priors (a fixed clustered set when cls is weak, so it
misses every GT); (2) predicted lanes were dropped by `row[1] < 0.5`, treating
the cls LOGIT as a 0/1 validity flag. Both are METRIC-only changes, so the
corrected ranking needs only a re-val over the saved checkpoints.

Run this script ONCE to (re)generate the notebook; keep it in version control.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [text]}


INTRO = """\
# NB100 - curveIoU re-validation of the NB98 ablation

Re-scores the 7 NB98 ablation checkpoints with the **fixed** curveIoU metric.
**No retraining** - each row loads its saved `best.pt` and runs a validation
pass only (~minutes/row), then we rank by the corrected `curveIoU`.

| Run | Technique |
|---|---|
| `abl_baseline`  | control |
| `abl_asym_lr`   | backbone LR x0.1 |
| `abl_gca_floor` | lane GCA gate floor 0.3 |
| `abl_det_decay` | det-loss x0.5 @ ep10 |
| `abl_freeze`    | freeze trunk @ ep10 |
| `abl_pcgrad`    | PCGrad gradient surgery |
| `abl_aux_seg`   | training-only dense lane-seg aux |

**Requires** each row's `best.pt` on Drive at
`EcoCAR/training_runs/checkpoints/<row>/best.pt` (the per-epoch drive-sync
saved them). If a checkpoint is missing, that row is skipped with a note -
re-run its NB98 training cell to regenerate it (the live curveIoU will be
correct there too, since the fix is in the imported metric module).
"""

CELL1_MD = "### Cell 1: Mount Drive + locate the repo"
CELL1 = '''\
import os, sys, subprocess, re
from pathlib import Path

if not Path('/content/drive/MyDrive').exists():
    try:
        from google.colab import drive
        print('[setup] mounting Drive...', flush=True)
        drive.mount('/content/drive', force_remount=False)
    except ImportError:
        pass

# clrkd (imported transitively by MTDETR during re-val) needs `addict`. In the
# training notebooks `mmcv` pulls it in; a FRESH re-val session lacks it -> the
# NB100 error `ModuleNotFoundError: No module named 'addict'`. Install the clrkd
# deps up front so the re-val SUBPROCESS (same Python env) can import MTDETR.
for _pkg in ('addict', 'yapf'):
    try:
        __import__(_pkg)
    except ImportError:
        print(f'[setup] installing {_pkg} ...', flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _pkg])
try:
    import mmcv  # noqa: F401
except ImportError:
    print('[setup] installing mmcv (clrkd dep) ...', flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv'])

def find_migration_root():
    cands = []
    cur = Path.cwd()
    for p in [cur, *cur.parents]:
        if p.name == 'rmt_ppad_migration':
            cands.append(p)
        cands.append(p / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration')
    for base in ('/content/drive/MyDrive/EcoCAR', '/content/drive/MyDrive', '/content'):
        cands.append(Path(base) / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration')
    for b in ('/content/drive/MyDrive/EcoCAR', '/content/drive/MyDrive'):
        bp = Path(b)
        if bp.exists():
            try:
                cands += list(bp.glob('*/yolop_vehicle_lane/stage2/rmt_ppad_migration'))
            except Exception:
                pass
    for c in cands:
        if (c / 'P8_train' / 'scripts' / 'revalidate_curveiou.py').exists():
            return c
    raise FileNotFoundError(
        'Could not find rmt_ppad_migration/ with P8_train/scripts/'
        'revalidate_curveiou.py. Re-sync the repo to Drive under '
        'EcoCAR/yolop_vehicle_lane/.')

MIG = find_migration_root()
REVAL = MIG / 'P8_train' / 'scripts' / 'revalidate_curveiou.py'
MODEL = MIG / 'vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml'
DATA  = MIG / 'vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml'
CKPT  = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints')
print('[ok] migration root:', MIG)
print('[ok] reval script   :', REVAL.exists())
print('[ok] checkpoints dir :', CKPT, '->', 'EXISTS' if CKPT.exists() else 'MISSING')
'''

CELL2_MD = """\
### Cell 2: Sanity-check the metric fix before spending time (CPU, instant)

Confirms the imported curve-F1 is the fixed broad-pool version: a synthetic
case where the top-8 priors miss the GT but a lower-ranked prior matches must
give curveIoU>0. If this fails, re-sync Drive (the metric module is stale).
"""
CELL2 = '''\
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('lcf1', MIG / 'P7_validator' / 'tools' / 'lane_curve_f1.py')
lcf1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lcf1)
try:
    import cv2  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python-headless'])

N, NS = 78, 71
def gtr(a0, L, a, b):
    r = np.full(N, -1e5); r[0]=0; r[1]=1; r[2]=a0; r[5]=float(L)
    st=int(round(a0*NS)); r[6+st:6+st+L]=np.linspace(a,b,L); return r
def prr(a0, L, a, b, p):
    r = np.full(N, -1e5); r[0]=0; r[1]=p; r[2]=a0; r[5]=L/NS
    st=int(round(a0*NS)); r[6+st:6+st+L]=np.linspace(a,b,L)/639.0; return r
gt = [gtr(0.0,40,420,440)]
rows = np.array([prr(0.4,11,85,95,9.0) for _ in range(8)] + [prr(0.0,40,422,442,-3.0) for _ in range(4)])
t8 = lcf1.lane_curve_tp_fp_fn(rows, np.array(gt), curveiou_pool=8)
t64 = lcf1.lane_curve_tp_fp_fn(rows, np.array(gt))
print(f'pool=8  curveIoU={t8[3]/max(1,t8[4]):.4f}  (old behavior, expect 0)')
print(f'pool=64 curveIoU={t64[3]/max(1,t64[4]):.4f}  (fixed, expect >0.5)')
assert t64[3]/max(1,t64[4]) > 0.5, 'METRIC IS STALE -> re-sync Drive (lane_curve_f1.py not updated)'
print('[ok] fixed curveIoU metric is in place')
'''

CELL_PREP_MD = """\
### Cell 2b: Extract the VAL subset to /content/ + point the data YAML at it

`model.val()` reads the images from `/content/bdd_subset_10k/`, but Colab wipes
`/content/` between sessions -- so a fresh NB100 run has no dataset and val dies
with `missing path '/content/bdd_subset_10k/images/val2017'`. This cell
re-extracts the SAME subset NB98 used (idempotent: skips if already present) and
rewrites the data YAML's `path` / `lane_targets_root` to the /content/ copy.
Mirrors NB98 Cells 2+4. (~80s the first time, then instant.)
"""
CELL_PREP = '''\
PREP = MIG / 'extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py'
SUBSET = Path('/content/bdd_subset_10k')
req = [Path('/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'),
       Path('/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/BDD_detection_labels.zip'),
       Path('/content/drive/MyDrive/EcoCAR/datasets/lane_targets_clr_v1_polyline.tar.gz')]
missing = [str(p) for p in req if not p.exists()]
if missing:
    raise FileNotFoundError('Missing Drive inputs:\\n  ' + '\\n  '.join(missing))

if (SUBSET / 'images/val2017').exists() and any((SUBSET / 'images/val2017').iterdir()):
    print(f'[ok] subset already at {SUBSET}; skipping extraction')
else:
    cmd = [sys.executable, '-u', str(PREP), '--out-root', str(SUBSET),
           '--n-train', '10000', '--n-val', '2000', '--seed', '89']
    print('  ', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError('subset prep failed - see streamed output above.')

# Point the data YAML at the /content/ copy; REMOVE drivable_masks_root (the
# NB98 collate bug -- mixed 1-/2-channel masks). Mirrors NB98 Cell 4.
def _yset(yaml_path, field, value):
    txt = yaml_path.read_text(encoding='utf-8'); line = f'{field}: {value}'
    if line in txt: return
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f'{field}:'): lines[i] = line; break
    else: lines.append(line)
    yaml_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8'); print(f'  [yaml] {field} -> {value}')

def _yunset(yaml_path, field):
    txt = yaml_path.read_text(encoding='utf-8')
    keep = [ln for ln in txt.splitlines() if not ln.strip().startswith(f'{field}:')]
    new = '\\n'.join(keep) + '\\n'
    if new != txt: yaml_path.write_text(new, encoding='utf-8'); print(f'  [yaml] removed {field}')

_yset(DATA, 'path', '/content/bdd_subset_10k')
_yset(DATA, 'lane_targets_root', '/content/bdd_subset_10k/lane_targets')
_yunset(DATA, 'drivable_masks_root')

for lab, d in (('images/val', SUBSET/'images/val2017'),
               ('lane_tgt/val', SUBSET/'lane_targets/val2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {lab:16s} {n}')
assert (SUBSET/'images/val2017').exists(), 'val images still missing after prep'
print('[ok] val subset ready + YAML configured')
'''

CELL3_MD = """\
### Cell 3: Re-validate every checkpoint + print the corrected ranking

For each row: load `best.pt`, run one validation pass (the fixed curveIoU is
computed in the validator), parse the `curveIoU` it prints. ~minutes/row.
"""
CELL3 = '''\
ROWS = ['abl_baseline','abl_asym_lr','abl_gca_floor','abl_det_decay',
        'abl_freeze','abl_pcgrad','abl_aux_seg']
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)
results = {}

for name in ROWS:
    bp = CKPT / name / 'best.pt'
    if not bp.exists():
        print(f'[skip] {name}: no best.pt at {bp} -> re-run its NB98 training cell')
        results[name] = None
        continue
    cmd = [sys.executable, '-u', str(REVAL), '--weights', str(bp),
           '--model-yaml', str(MODEL), '--data-yaml', str(DATA), '--name', f'reval_{name}']
    if name == 'abl_aux_seg':
        cmd.append('--use-aux-seg')
    print(f'\\n===== re-val {name} =====', flush=True)
    cap = []
    with open(os.path.join(LOG_DIR, f'NB100_reval_{name}.log'), 'w') as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True); logf.write(line); cap.append(line)
        proc.wait()
    txt = ''.join(cap)
    # The numbers live in the validator's LOG lines, NOT in model.val()'s
    # results_dict (which only carries detection keys -> the script's DONE
    # line shows None). Parse the captured subprocess stdout directly:
    #   [lane-f1] curve F1=0.1700 ... mean-best-IoU=0.5171
    #   lane  pixacc 0.953 subacc 0.587 IoU 0.047 mIoU 0.5
    def _grab(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    results[name] = {
        'curveIoU': _grab(r'mean-best-IoU=([0-9.]+)'),
        'F1':       _grab(r'curve F1=([0-9.]+)'),
        'pixelIoU': _grab(r'IoU\\s+([0-9.]+)\\s+mIoU'),
        'subacc':   _grab(r'subacc\\s+([0-9.]+)'),
    }

print('\\n' + '='*78)
print('CORRECTED ablation ranking (re-val with fixed curveIoU metric)')
print('='*78)
print(f'{"row":16s} {"curveIoU":>9} {"F1@0.5":>8} {"pixelIoU":>9} {"subAcc":>8}')
print('-'*54)
# Rank by F1@0.5 (the discriminative metric; curveIoU=max-over-pool is nearly
# flat across rows and a weak discriminator).
def _key(kv):
    d = kv[1] or {}
    return (d.get('F1') if d.get('F1') is not None else -1)
for k, d in sorted(results.items(), key=_key, reverse=True):
    d = d or {}
    def _f(x): return f'{x:.4f}' if isinstance(x, float) else '   -   '
    print(f'{k:16s} {_f(d.get("curveIoU")):>9} {_f(d.get("F1")):>8} '
          f'{_f(d.get("pixelIoU")):>9} {_f(d.get("subacc")):>8}')
base = (results.get('abl_baseline') or {}).get('F1')
print(f'\\nbaseline F1@0.5 = {base}.  Winner = top non-baseline row that beats it on F1 AND pixelIoU.')
print('(curveIoU = max-match over a 64-prior pool: near-saturated, low discrimination here.)')
'''


def build():
    nb = {
        "cells": [
            md(INTRO),
            md(CELL1_MD), code(CELL1),
            md(CELL_PREP_MD), code(CELL_PREP),
            md(CELL2_MD), code(CELL2),
            md(CELL3_MD), code(CELL3),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_100_curveiou_revalidation.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
