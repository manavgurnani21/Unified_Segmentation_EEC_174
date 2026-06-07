"""One-shot builder for stage2_notebook_94_hungarian_probe.ipynb.

Phase-1 go/no-go probe: does Hungarian 1-to-1 matching unfreeze the
decoded lane IoU that dynamic-k leaves stuck at 0.0718?

Runs the SAME polyline head on the SAME 10k subset twice:
  Row A: --lane-match hungarian   (the fix)
  Row B: --lane-match dynamic_k   (the control = old frozen behavior)

PASS = Row A's IoU(lane) climbs past 0.10 and rises across epochs, and
its scoreStd column grows away from 0, while Row B stays frozen.
Re-run this builder to regenerate the .ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": [t]}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [t]}


INTRO = """\
# NB94 - Phase 1 probe: Hungarian matching unfreezes lane IoU?

The NB88/89/92 runs showed decoded IoU(lane) FROZEN at 0.0718 because the
lane cls head collapses to a uniform 0.5 sigmoid under dynamic-k matching
(a prior is labeled positive in some batches, negative in others). The
~70-experiment prior research (exp01-71) proved the fix is **Hungarian
1-to-1 matching**, which gives every matched prior a deterministic
positive label so the cls can separate.

This notebook runs the SAME polyline head on the SAME 10k subset twice:

| Row | --lane-match | expectation |
|-----|--------------|-------------|
| A   | `hungarian` (the fix) | IoU(lane) climbs past 0.10, scoreStd grows |
| B   | `dynamic_k` (control) | IoU(lane) frozen ~0.07, scoreStd ~0 (dead) |

New diagnostics added this round (visible in the per-epoch table + log):
- **IoU now uses top-N rasterization (no threshold)** so it tracks the
  cls ranking instead of freezing behind a 0.5 cutoff.
- **`[lane-score]` histogram** printed each val epoch - eyeball whether
  the 192 priors are still clustered at 0.5.
- **`scoreStd` / `scoreSprd` columns** + an automatic
  `[dead-lane-monitor]` warning if the cls stays collapsed.

PASS criteria: Row A IoU(lane) > 0.10 and rising by epoch 30; Row A
scoreStd visibly > Row B scoreStd.
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
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    import mmcv  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mmcv'])
# scipy is needed for Hungarian matching; present on Colab but assert it.
import scipy.optimize  # noqa: F401
print('[ok] env ready (scipy present for hungarian matching)')
'''

CELL2_MD = "### Cell 2: Build / reuse the 10k polyline subset on /content/"
CELL2 = '''\
import sys, subprocess
from pathlib import Path
PREP = 'stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py'
SUB = Path('/content/bdd_subset_10k')
if (SUB / 'prep_summary.json').exists():
    print(f'[ok] subset already prepared at {SUB}; skipping')
else:
    subprocess.check_call([sys.executable, '-u', PREP, '--out-root', str(SUB),
                           '--n-train', '10000', '--n-val', '2000', '--seed', '89'])
for lbl, d in (('images/train', SUB/'images/train2017'),
               ('lane_tgt/train', SUB/'lane_targets/train2017'),
               ('lane_tgt/val',   SUB/'lane_targets/val2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {lbl:16s} {n}')
'''

CELL3_MD = "### Cell 3: Helper - `launch_probe(match)` (polyline head, 30 ep)"
CELL3 = '''\
import os, sys, subprocess, shutil
from pathlib import Path
REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from stage2.scripts.notebook_utils import run_streaming

LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)
VDATA = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/datasets'
VMOD  = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr'
MODEL = VMOD / 'rtdetr-l_bdd_clr_lane.yaml'         # polyline head
DATA  = VDATA / 'BDD_lane_only_10k.yaml'            # 10k polyline targets
TRAIN = 'stage2/rmt_ppad_migration/P8_train/scripts/train_lane_only.py'

# Make sure the 10k data YAML points at the subset on /content/.
def _ensure(yaml_path, field, val):
    t = yaml_path.read_text(encoding='utf-8'); line = f'{field}: {val}'
    if line in t: return
    ls = t.splitlines()
    for i, l in enumerate(ls):
        if l.strip().startswith(field + ':'):
            ls[i] = line; yaml_path.write_text(chr(10).join(ls)+chr(10), encoding='utf-8')
            print(f'  [yaml] {yaml_path.name}: {field} -> {val}'); return
_ensure(DATA, 'path', '/content/bdd_subset_10k')
_ensure(DATA, 'lane_targets_root', '/content/bdd_subset_10k/lane_targets')

def launch_probe(match: str, epochs: int = 30, batch: int = 32, lr0: float = 4e-4,
                 fresh: bool = True) -> bool:
    name = f'probe_polyline_{match}'
    rd = Path('/content/runs') / name
    if fresh and rd.exists():
        print(f'[fresh] wiping {rd}'); shutil.rmtree(rd, ignore_errors=True)
    cmd = [sys.executable, '-u', TRAIN, '--mode', 'full', '--name', name,
           '--project', '/content/runs', '--model-yaml', str(MODEL),
           '--data-yaml', str(DATA), '--device', '0', '--workers', '8',
           '--save-period', '10', '--batch', str(batch), '--epochs', str(epochs),
           '--lr0', str(lr0), '--lane-match', match]
    log = os.path.join(LOG_DIR, f'NB94_{name}.log')
    print(f'\\n=== launching {name} (match={match}, ep={epochs}, batch={batch}) ===\\n', flush=True)
    rc = run_streaming(cmd, log_path=log, check=False)
    print(f'{name} rc={rc}; log -> {log}')
    return rc == 0

print('[ready] launch_probe("hungarian") then launch_probe("dynamic_k")')
'''

CELL4_MD = "### Cell 4: Row A - Hungarian matching (the fix)"
CELL4 = '''\
ok = launch_probe('hungarian', epochs=30, batch=32, lr0=4e-4, fresh=True)
print('row A done' if ok else 'row A incomplete')
'''

CELL5_MD = "### Cell 5: Row B - dynamic_k control (expect frozen IoU)"
CELL5 = '''\
ok = launch_probe('dynamic_k', epochs=30, batch=32, lr0=4e-4, fresh=True)
print('row B done' if ok else 'row B incomplete')
'''

CELL6_MD = "### Cell 6: Compare IoU + scoreStd trajectories (the verdict)"
CELL6 = '''\
import csv
from pathlib import Path
def load(name):
    p = Path('/content/runs') / name / 'results.csv'
    if not p.exists(): return []
    with p.open() as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
def col(rows, key):
    out = []
    for r in rows:
        try: out.append(float(r.get(key, '')))
        except ValueError: out.append(None)
    return out

print(f'{"ep":>3} | {"HUN IoU":>9} {"HUN std":>9} | {"DYN IoU":>9} {"DYN std":>9}')
print('-' * 52)
h = load('probe_polyline_hungarian'); d = load('probe_polyline_dynamic_k')
hi, hs = col(h, 'metrics/IoU(lane)'), col(h, 'metrics/lane_score_std(lane)')
di, ds = col(d, 'metrics/IoU(lane)'), col(d, 'metrics/lane_score_std(lane)')
n = max(len(h), len(d))
def g(a, i):
    return f'{a[i]:.4f}' if i < len(a) and a[i] is not None else '   -   '
for i in range(n):
    print(f'{i+1:>3} | {g(hi,i):>9} {g(hs,i):>9} | {g(di,i):>9} {g(ds,i):>9}')

# Verdict
def last(a):
    vals = [x for x in a if x is not None]
    return vals[-1] if vals else None
hl, dl = last(hi), last(di)
print()
if hl is not None and hl > 0.10 and (dl is None or hl > dl + 0.02):
    print(f'[VERDICT] PASS - Hungarian IoU={hl:.4f} > 0.10 and beats dynamic_k '
          f'({dl}). Phase 1 fix works; proceed to Phase 4 ablation.')
else:
    print(f'[VERDICT] INCONCLUSIVE - Hungarian IoU={hl}, dynamic_k IoU={dl}. '
          f'If both frozen, the fix is insufficient; escalate to the K=64 '
          f'query head (Exp2RR) per PLAN_AFTER_NB88_89_92.md.')
'''


def build():
    nb = {
        "cells": [md(INTRO),
                  md("### Cell 1: Mount + env"), code(CELL1),
                  md(CELL2_MD), code(CELL2),
                  md(CELL3_MD), code(CELL3),
                  md(CELL4_MD), code(CELL4),
                  md(CELL5_MD), code(CELL5),
                  md(CELL6_MD), code(CELL6)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_94_hungarian_probe.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
