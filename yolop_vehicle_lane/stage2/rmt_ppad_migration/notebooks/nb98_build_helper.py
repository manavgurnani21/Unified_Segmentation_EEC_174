"""One-shot builder for stage2_notebook_98_mtl_negative_transfer.ipynb.

Run this script ONCE to (re)generate the notebook. Self-contained per the
Colab /content/ isolation rule: prep + drivable extraction + aux sanity
check + ablation rows + aggregation all live in ONE notebook session.
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
# NB98 - MTL negative-transfer ablation suite (+ aux dense seg)

Runs the anti-negative-transfer techniques head-to-head on the 10k subset,
20 epochs each, and judges them by **`metrics/lane_curveIoU(lane)`** (the
un-saturated metric - `lane_f1` will likely stay 0). A technique wins if
curveIoU stays stable / rising past epoch 10 vs `abl_baseline`.

| Row | Flag | Technique |
|---|---|---|
| `abl_baseline`  | (none)                | control: peak-then-erode |
| `abl_asym_lr`   | `--backbone-lr-mult 0.1` | shared trunk at 0.1x LR |
| `abl_gca_floor` | `--seg-gate-floor 0.3`   | force lane GCA decoupling |
| `abl_det_decay` | `--det-decay-epoch 10`   | halve det loss at ep10 |
| `abl_freeze`    | `--freeze-trunk-after 10`| freeze trunk at ep10 |
| `abl_pcgrad`    | `--pcgrad`               | gradient surgery |
| `abl_aux_seg`   | `--use-aux-seg`          | training-only dense LANE-seg aux (drivable deferred, see Cell 3) |

**How to run:** Cells 1-4 prep the data (idempotent; re-run them at the
start of EVERY new Colab session because /content/ is wiped between
sessions). **Cell 5 is a GATE** - it proves the aux heads route correctly
(train-only, dropped at eval) and sets `AUX_ROUTING_OK`; the `abl_aux_seg`
row (Cell 13) refuses to launch unless it passed, so no GPU time is wasted
on broken aux wiring. The other six rows are independent of Cell 5. Then
run whichever row cells (7-13) you have GPU time for - each ~1.5 h,
resume-safe. Cell 14 aggregates whatever has finished.

> **Fixed (2026-05-31):** two bugs from the first run are resolved here.
> (1) Cell 4 used to wire `drivable_masks_root` into the shared data YAML,
> but only ~460/12000 subset images have drivable masks, so the loader
> emitted mixed 1-/2-channel `merge_mask` tensors and `collate_fn` crashed
> on EVERY row (`torch.stack ... [1,640,640] vs [2,640,640]`). Cell 4 now
> actively *removes* that key -> the 6 core rows run on the clean lane-only
> config. (2) The Cell-5 routing test crashed with `KeyError 'batch'`
> because it ran the full decoder train-forward (which invokes the CLR lane
> head's loss path); it now exercises the aux module directly. The
> `abl_aux_seg` row therefore runs as **dense lane-seg aux only** -
> drivable supervision is deferred until coverage is adequate.
"""

CELL1_MD = "### Cell 1: Mount Drive + locate sources"
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
print('[ok] env ready; cwd =', os.getcwd())
'''

CELL2_MD = """\
### Cell 2: Extract the 10k subset to /content/ (images + labels + lane targets)

Idempotent: skips if already prepared. MUST extract to /content/ (NOT Drive)
per the small-files rule.
"""
CELL2 = '''\
import sys, subprocess
from pathlib import Path

PREP = Path('stage2/rmt_ppad_migration/extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py')
SUBSET = Path('/content/bdd_subset_10k')
req = [Path('/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'),
       Path('/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/BDD_detection_labels.zip'),
       Path('/content/drive/MyDrive/EcoCAR/datasets/lane_targets_clr_v1_polyline.tar.gz')]
missing = [str(p) for p in req if not p.exists()]
if missing:
    raise FileNotFoundError('Missing Drive inputs:\\n  ' + '\\n  '.join(missing))

if (SUBSET / 'prep_summary.json').exists():
    print(f'[ok] subset already at {SUBSET}; skipping')
else:
    cmd = [sys.executable, '-u', str(PREP), '--out-root', str(SUBSET),
           '--n-train', '10000', '--n-val', '2000', '--seed', '89']
    print('  ', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError('subset prep failed - see streamed output above.')
for lab, d in (('images/train', SUBSET/'images/train2017'), ('images/val', SUBSET/'images/val2017'),
               ('lane_tgt/train', SUBSET/'lane_targets/train2017')):
    n = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f'  {lab:16s} {n}')
'''

CELL3_MD = """\
### Cell 3: (OPTIONAL / deferred) Extract drivable masks for the subset

> **Deferred.** On the first run this matched only ~460 of 12000 subset
> images (`bdd100k_seg_maps.zip` covers a different split than the curve
> subset). That partial coverage is what broke collation, and it is too
> sparse to be useful drivable supervision anyway, so Cell 4 does **not**
> wire drivable into the data YAML and the `abl_aux_seg` row runs as
> **lane-seg dense aux only**. This cell is kept for when full-coverage
> drivable masks are available; running it now is harmless but optional.

Pulls only the masks whose stem matches a subset image from
`EcoCAR/downloads/bdd100k_seg_maps.zip` and writes a binary drivable mask to
`/content/bdd_subset_10k/drivable_masks/{split}/<stem>.png`. Self-inspecting:
prints the zip layout + a value histogram so a 0-match can be diagnosed.
"""
CELL3 = '''\
import sys, subprocess
from pathlib import Path

EXTRACT = Path('stage2/rmt_ppad_migration/aux_seg/tools/extract_drivable_subset.py')
SEG_ZIP = Path('/content/drive/MyDrive/EcoCAR/downloads/bdd100k_seg_maps.zip')
SUBSET = Path('/content/bdd_subset_10k')

if not SEG_ZIP.exists():
    print(f'[WARN] {SEG_ZIP} not found -> aux row will be LANE-ONLY aux.')
elif (SUBSET / 'drivable_masks/train2017').exists() and \\
     any((SUBSET / 'drivable_masks/train2017').iterdir()):
    print('[ok] drivable masks already extracted; skipping')
else:
    cmd = [sys.executable, '-u', str(EXTRACT), '--seg-zip', str(SEG_ZIP),
           '--subset-root', str(SUBSET)]
    print('  ', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end='', flush=True)
    rc = p.wait()
    if rc != 0:
        print(f'[WARN] extractor rc={rc}: aux row will fall back to lane-only aux.')

# To diagnose a 0-match:
# !python stage2/rmt_ppad_migration/aux_seg/tools/extract_drivable_subset.py \\
#     --seg-zip /content/drive/MyDrive/EcoCAR/downloads/bdd100k_seg_maps.zip --inspect-only
'''

CELL4_MD = """\
### Cell 4: Point the data YAML at the /content/ subset (+ drivable masks)
"""
CELL4 = '''\
from pathlib import Path

MIG = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration'
DATA_YAML = MIG / 'vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml'

def _set(yaml_path, field, value):
    txt = yaml_path.read_text(encoding='utf-8')
    line = f'{field}: {value}'
    if line in txt: return
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f'{field}:'):
            lines[i] = line; break
    else:
        lines.append(line)
    yaml_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
    print(f'  [yaml] {field} -> {value}')

def _unset(yaml_path, field):
    """Remove a field line entirely (idempotent)."""
    txt = yaml_path.read_text(encoding='utf-8')
    lines = [ln for ln in txt.splitlines() if not ln.strip().startswith(f'{field}:')]
    new = '\\n'.join(lines) + '\\n'
    if new != txt:
        yaml_path.write_text(new, encoding='utf-8')
        print(f'  [yaml] removed {field}')

_set(DATA_YAML, 'path', '/content/bdd_subset_10k')
_set(DATA_YAML, 'lane_targets_root', '/content/bdd_subset_10k/lane_targets')
# CRITICAL: do NOT set drivable_masks_root on the shared YAML. Drivable
# coverage on this subset is only ~460/12000 images, so the loader emits a
# 2-channel merge_mask ([drivable,lane]) for those and 1-channel ([lane]) for
# the rest -> collate_fn's torch.stack fails for EVERY row ("stack expects
# each tensor to be equal size, got [1,640,640] vs [2,640,640]"). We ACTIVELY
# REMOVE it (a previous notebook run may have written it into the YAML on
# Drive). The aux row uses lane-seg dense aux only; drivable is deferred until
# coverage is adequate (see Cell 3).
_unset(DATA_YAML, 'drivable_masks_root')
print('[ok] data yaml configured (lane-only, no drivable):', DATA_YAML.name)
'''

CELL5_MD = """\
### Cell 5: Aux routing GATE (no training, ~1 min) - run BEFORE the aux row

Proves the auxiliary dense heads RUN in train mode and are fully DROPPED in
eval mode (zero inference overhead). This is a **GATE**, not just a print:
the `abl_aux_seg` row (Cell 13) refuses to launch unless this cell has set
`AUX_ROUTING_OK = True`, so no GPU hours are spent on a broken aux wiring.

Data-independent (needs only the model YAML + torch), so you can run it
right after Cell 1 if you just want to validate the aux plumbing - it does
NOT require the subset prep in Cells 2-4. The other six ablation rows do not
depend on aux and are not gated by this.
"""
CELL5 = '''\
import os, sys, subprocess
from pathlib import Path
MIG = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration'
env = dict(os.environ, USE_AUX_SEG='1', AUX_SEG_CLASSES='2', PYTHONIOENCODING='utf-8')
cmd = [sys.executable, '-u', str(MIG / 'aux_seg/test_aux_routing.py')]
print('  ', ' '.join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
for line in p.stdout: print(line, end='', flush=True)
_rc = p.wait()
# Persist a notebook-global so the aux row can refuse to launch on failure.
# We deliberately do NOT raise: a non-zero rc must NOT block the other six
# (non-aux) rows, which are independent of the aux plumbing.
AUX_ROUTING_OK = (_rc == 0)
print(f'\\n[routing test rc = {_rc}] -> AUX_ROUTING_OK = {AUX_ROUTING_OK}')
if AUX_ROUTING_OK:
    print('[GATE PASS] aux runs in train, drops at eval -> abl_aux_seg row is cleared to launch.')
else:
    print('[GATE FAIL] do NOT run the abl_aux_seg row until this passes. '
          'Fix the routing first; the other 6 rows are unaffected.')
'''

CELL6_MD = """\
### Cell 6: `launch_ablation(name, *flags)` helper

Each row = baseline config + exactly one technique flag. Resume-safe
(re-run the same cell to continue). best.pt/last.pt + full_train.log sync
to Drive every epoch via train_lane_only's drive-sync callback.
"""
CELL6 = '''\
import os, sys
from pathlib import Path
from stage2.scripts.notebook_utils import run_streaming

MIG = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration'
TRAIN = MIG / 'P8_train/scripts/train_lane_only.py'
MODEL = MIG / 'vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml'
DATA  = MIG / 'vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml'
PROJECT = '/content/runs/mtl_ablation'
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Shared config = current NB97 recipe, so the ONLY diff between rows is the
# technique under test. patience 99 > 20 epochs => no early stop (we want to
# SEE the full post-ep10 trajectory).
COMMON = ['--mode', 'full', '--model-yaml', str(MODEL), '--data-yaml', str(DATA),
          '--project', PROJECT, '--device', '0', '--epochs', '20', '--batch', '32',
          '--lr0', '4e-4', '--lane-weights', 'clrkd', '--diff-clamp', '100',
          '--lane-match', 'hungarian', '--fliplr', '0.5', '--weight-decay', '0.05',
          '--val-period', '1', '--patience', '99', '--save-period', '20']

def launch_ablation(name, *flags):
    cmd = [sys.executable, '-u', str(TRAIN)] + COMMON + ['--name', name] + list(flags)
    log = os.path.join(LOG_DIR, f'NB98_{name}.log')
    print(f'\\n=== {name}  flags={list(flags)} ===\\n', flush=True)
    rc = run_streaming(cmd, log_path=log, check=False)
    print(f'[{name}] rc={rc}  (watch metrics/lane_curveIoU(lane))')
    return rc == 0

print('[ready] run the row cells below in any order; each ~1.5 h.')
'''

# The aux row guards on the Cell-5 gate. Defined as a real multi-line block
# (triple-quoted = actual newlines) so it executes as statements - an earlier
# version used literal "\\n" which collapsed the whole thing into one comment
# line that silently did nothing.
AUX_ROW_CALL = '''\
# GATE: requires the Cell-5 aux routing check to have passed.
if not globals().get('AUX_ROUTING_OK', False):
    raise RuntimeError(
        'Run Cell 5 (aux routing GATE) first and ensure it PASSES before '
        'launching the aux row - refusing to spend ~1.5 GPU-h on unproven '
        'aux wiring.')
launch_ablation('abl_aux_seg', '--use-aux-seg', '--aux-seg-classes', '2',
                '--aux-drivable-weight', '0.5', '--aux-lane-weight', '0.5')'''

ROWS = [
    ('abl_baseline', "Row 0: baseline (no technique) - the curve to beat", "launch_ablation('abl_baseline')"),
    ('abl_asym_lr', "Row 1: asymmetric LR (backbone x0.1)", "launch_ablation('abl_asym_lr', '--backbone-lr-mult', '0.1')"),
    ('abl_gca_floor', "Row 2: strengthen GCA (lane gate floor 0.3)", "launch_ablation('abl_gca_floor', '--seg-gate-floor', '0.3')"),
    ('abl_det_decay', "Row 3: dynamic det-loss decay (x0.5 @ ep10)", "launch_ablation('abl_det_decay', '--det-decay-epoch', '10', '--det-decay-factor', '0.5')"),
    ('abl_freeze', "Row 4: freeze trunk @ ep10", "launch_ablation('abl_freeze', '--freeze-trunk-after', '10')"),
    ('abl_pcgrad', "Row 5: PCGrad gradient surgery (~2x backward)", "launch_ablation('abl_pcgrad', '--pcgrad')"),
    ('abl_aux_seg', "Row 6: aux dense seg (drivable+lane, training-only) - GATED by Cell 5",
     AUX_ROW_CALL),
]

CELL_AGG_MD = """\
### Cell 14: Aggregate - compare curveIoU across whatever rows finished

For each row, prints the curveIoU at ep1 / ep10 / final + the post-ep10
delta. A technique WINS if its post-ep10 delta is >= 0 (stable/rising)
while baseline's is negative (the decline).
"""
CELL_AGG = '''\
import csv
from pathlib import Path

PROJECT = Path('/content/runs/mtl_ablation')
ROWS = ['abl_baseline','abl_asym_lr','abl_gca_floor','abl_det_decay',
        'abl_freeze','abl_pcgrad','abl_aux_seg']

def _col(rows, *keys):
    out = []
    for r in rows:
        v = None
        for k in keys:
            for ck in r:
                if ck.strip().lower() == k.lower():
                    try: v = float(r[ck])
                    except: pass
        out.append(v)
    return out

print(f'{"row":16s} {"cIoU@1":>8} {"cIoU@10":>8} {"cIoU@end":>9} {"post10 d":>9}  verdict')
print('-'*64)
for name in ROWS:
    csvp = PROJECT / name / 'results.csv'
    if not csvp.exists():
        print(f'{name:16s} {"(not run)":>8}'); continue
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(csvp))]
    ci = _col(rows, 'metrics/lane_curveIoU(lane)')
    ci = [c for c in ci if c is not None]
    if len(ci) < 2:
        print(f'{name:16s} {"(no cIoU)":>8} - re-sync? curveIoU is a new metric'); continue
    e1, e10 = ci[0], ci[min(9, len(ci)-1)]
    eend = ci[-1]
    delta = eend - e10
    verdict = 'RISING/stable' if delta >= -0.002 else 'declines'
    print(f'{name:16s} {e1:8.4f} {e10:8.4f} {eend:9.4f} {delta:+9.4f}  {verdict}')
print('\\nWinner = best post-ep10 delta (>=0) AND highest cIoU@end.')
'''


def build():
    cells = [md(INTRO), md(CELL1_MD), code(CELL1), md(CELL2_MD), code(CELL2),
             md(CELL3_MD), code(CELL3), md(CELL4_MD), code(CELL4),
             md(CELL5_MD), code(CELL5), md(CELL6_MD), code(CELL6)]
    for i, (name, title, call) in enumerate(ROWS, start=7):
        cells.append(md(f"### Cell {i}: {title}"))
        cells.append(code(call + "\n"))
    cells.append(md(CELL_AGG_MD)); cells.append(code(CELL_AGG))
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).resolve().parent / 'stage2_notebook_98_mtl_negative_transfer.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
