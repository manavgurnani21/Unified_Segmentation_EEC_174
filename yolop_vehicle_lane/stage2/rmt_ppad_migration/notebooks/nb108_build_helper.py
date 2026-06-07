"""One-shot builder for stage2_notebook_108_laneiou_resume_yreweight.ipynb.

NB108 = RESUME of the NB107 `laneiou_both_full` run (stopped at epoch 18 of 120)
with ONE new lever turned on to fix the diagnosed defect.

WHY (see results/NB107_IOU_STUCK_PLAN.md):
  NB107 climbed beautifully on the faithful lane metrics -- curve-F1 0.001->0.608,
  curveIoU 0.41->0.735, mAP50 0.14->0.82 -- but the user flagged "lane IoU stuck"
  (pixel-IoU(lane) flat ~0.05). That pixel-IoU is the SATURATING thin-line mask
  metric we already moved the fitness off of; it is the wrong meter. The ONE real
  signal inside it is `length_mean` SHRINKING 0.155->0.094: the model draws
  increasingly SHORT lanes (covers the easy mid-section, abandons the near/far
  extremities), so the drawn mask stays tiny even as the curves get more accurate.

THE FIX (resume, do NOT restart -- user explicit):
  Turn ON the already-built-but-OFF S1.4 near-field reweight: `--lane-y-reweight
  near` puts 1.85x weight on the near (bottom) rows where length is abandoned, so
  shortening becomes costly there. It composes with the angle-aware LaneIoU loss
  (lane_iou_loss_fn), threaded via the LANE_Y_REWEIGHT env var so it DOES take
  effect on resume even though ultralytics reloads the checkpoint's saved args.

EVERYTHING ELSE IS IDENTICAL TO NB107 (same model YAML, 70k/10k prep, aux+gca
recipe, LaneIoU both, tau 0.4, curvef1 fitness). The only two deltas vs NB107:
  (1) restore last.pt from Drive, then  (2) `--resume auto --lane-y-reweight near`.

This builder IMPORTS nb107_build_helper and reuses its dataset-prep cells verbatim
(no copy/drift); it overrides only INTRO, adds the restore cell, and swaps Cell 4.

Run this script ONCE to (re)generate the notebook; keep it in version control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb107_build_helper as nb107  # noqa: E402  (reuse its verbatim cells)

md, code = nb107.md, nb107.code


INTRO = """\
# NB108 - RESUME NB107 + near-field y-reweight (fix the length-shrink)

**Continues** the NB107 `laneiou_both_full` run from its last checkpoint (it
stopped at **epoch 18 / 120**) -- it does **NOT** restart from scratch.

## Why
NB107's faithful lane metrics were the best in the project and still climbing:

| metric | ep1 | ep18 | |
|---|---|---|---|
| lane curve-F1 | 0.001 | **0.608** | rising (cleared G2's 0.45) |
| curveIoU | 0.407 | **0.735** | rising (highest ever) |
| mAP50 (guardrail) | 0.14 | **0.823** | healthy |
| pixel-IoU(lane) | 0.048 | **0.054** | flat -- the "stuck" |
| **length_mean** | 0.155 | **0.094** | **shrinking** |

The "stuck lane IoU" is **pixel-IoU**, the saturating thin-line mask metric we
already moved the fitness off of (it sat at ~0.05 while F1 went 0.001->0.608 --
different things). The ONE real defect is **length_mean shrinking**: the model
draws shorter lanes, so the drawn mask stays tiny. That is the documented S1.4
length-shrink failure.

## The fix (two complementary levers, on resume)
The shrink lives in the predicted **`length` field**, which the xytl smooth-L1
supervises *symmetrically* and weakly (1/4 of a mixed-scale mean), so the model
freely shortens lanes to dodge x-error on hard rows. Two levers attack it:

1. **`--lane-len-hinge 2.0` (DIRECT, S1.4b)** -- an *asymmetric* hinge that
   penalizes ONLY too-short matched lanes: `(gt_len - pred_len)_+` normalized to
   [0,1]. This raises the cost of shortening head-on, on the matched (real) lanes.
2. **`--lane-y-reweight near` (INDIRECT, S1.4)** -- 1.85x weight on near-row
   x-accuracy in the LaneIoU loss; sharpens the near-field geometry the model
   abandons. (On its own this does NOT touch the length field -- that is why the
   hinge above was added.)

Both are threaded via env vars read at loss-construction, so they **apply on
resume** even though ultralytics reloads the checkpoint's saved hyperparams.
`--lane-len-hinge 2.0` is a starting estimate -- it is a tunable knob; raise to
3-4 if length does not recover, lower if curve-F1 dips. Everything else is
byte-identical to NB107.

## What to watch
- **pixel-IoU(lane) + mask_frac should now RISE** (more lane actually drawn) --
  these are the cleanest "is the lane longer" signals.
- **curve-F1 / curveIoU keep rising or hold** and **mAP50 stays >=0.48**.
- `length_mean` is a NOISY proxy (it averages all 192 priors, not just matched
  lanes), so weight pixel-IoU/mask_frac + F1 over it.
Re-judge the head on **curve-F1**, NOT pixel-IoU-as-a-target. If pixel-IoU/
mask_frac are still flat by ~ep28, raise `--lane-len-hinge`.

## How to run (fresh Colab session)
Cells 1-3 prep (mount, deps, extract 70k/10k to `/content/`, point the YAML) --
**identical to NB107, idempotent**. Cell 3b: optional visual sanity check of the
freshly re-extracted data. **Cell 3c: REQUIRED -- restores `last.pt` from Drive**
(`/content/` is wiped between sessions, so the checkpoint lives only on Drive).
Cell 4 relaunches with `--resume auto --lane-y-reweight near`. Cell 5 reads the
trajectory back.

> Per the Colab `/content/` isolation rule, the dataset MUST be re-extracted to
> `/content/` every new session; the checkpoint MUST be restored from Drive
> (Cell 3c) because `/content/runs/...` starts empty.
"""

CELL_RESTORE_MD = """\
### Cell 3c: Restore the NB107 checkpoint from Drive (REQUIRED for resume)

`/content/` is wiped between Colab sessions, so `/content/runs/laneiou_full/`
starts EMPTY -- but `drive_sync` mirrored `last.pt` / `best.pt` / `results.csv`
to Drive every epoch. This cell copies them back to the canonical local run dir
so `--resume auto` (Cell 4) finds `last.pt` at
`{project}/{name}/weights/last.pt`. It also pre-reads the checkpoint epoch so
you know where the resume starts BEFORE spending GPU.
"""
CELL_RESTORE = '''\
import shutil
from pathlib import Path

# drive_sync (epoch_callbacks.py) mirrors files FLAT into this dir each epoch:
#   last.pt, best.pt, results.csv, full_train.log, log_epoch_NNN.json
DRIVE_RUN = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints/laneiou_both_full')
LOCAL_RUN = Path('/content/runs/laneiou_full/laneiou_both_full')   # = PROJECT/ROW in Cell 4
(LOCAL_RUN / 'weights').mkdir(parents=True, exist_ok=True)

src_last = DRIVE_RUN / 'last.pt'
if not src_last.exists():
    listing = sorted(p.name for p in DRIVE_RUN.glob('*')) if DRIVE_RUN.exists() else ['<dir missing>']
    raise FileNotFoundError(
        f'No last.pt on Drive at {src_last}.\\n'
        'Cannot resume -- verify NB107 ran and drive_sync wrote the checkpoint.\\n'
        f'{DRIVE_RUN} contains: {listing[:20]}')

# last.pt -> the canonical local weights path so `--resume auto` resolves it.
dst_last = LOCAL_RUN / 'weights' / 'last.pt'
shutil.copy2(src_last, dst_last)
# best.pt + results.csv: continuity. (best_fitness ALSO lives inside last.pt, so
# resume tracking is correct even without best.pt; we keep the real best file
# and the prior results rows so the trajectory stays continuous.)
for fn in ('best.pt', 'results.csv'):
    s = DRIVE_RUN / fn
    if s.exists():
        dst = (LOCAL_RUN / 'weights' / fn) if fn.endswith('.pt') else (LOCAL_RUN / fn)
        shutil.copy2(s, dst)
print(f'[restore] copied last.pt ({src_last.stat().st_size/1e6:.1f} MB) -> {dst_last}')

# Pre-flight: report the checkpoint epoch (ultralytics ALSO prints the
# authoritative "Resuming training ... from epoch N to 120" line in Cell 4).
try:
    import torch
    ck = torch.load(dst_last, map_location='cpu', weights_only=False)
    print(f'[restore] ckpt epoch index = {ck.get("epoch")}   best_fitness = {ck.get("best_fitness")}')
    print('[restore] training will continue to epoch 120 (the original target).')
    del ck
except Exception as e:
    print(f'[restore] (could not pre-read epoch: {type(e).__name__}: {e}; '
          'Cell 4 prints the authoritative resume epoch.)')
'''

CELL4_MD = """\
### Cell 4: RESUME the run from last.pt + length-shrink fix

Identical to NB107's launch EXCEPT three deltas:
`--resume auto` (continue from the restored `last.pt`, epoch 19 -> 120),
`--lane-len-hinge 2.0` (DIRECT asymmetric too-short penalty on matched lanes),
and `--lane-y-reweight near` (INDIRECT near-field x-accuracy). All other flags --
aux_seg, gca_floor, LaneIoU both, tau 0.4, clrkd weights, fliplr/wd -- are
unchanged, and `LANE_FITNESS_METRIC=curvef1` keeps `best.pt` tracking curveIoU+F1.
The startup log line `[MTDETRDLoss] S1 levers: ... len_hinge_w=2.0` confirms the
hinge is live.

**Sanity-check the log:** it MUST print `Resuming training ... from epoch ~19 to
120 total epochs`. If it prints epoch 1, the resume did NOT take -- stop and
check Cell 3c before burning GPU.
"""
CELL4 = '''\
from stage2.scripts.notebook_utils import run_streaming
import os as _os_fit
_os_fit.environ['LANE_FITNESS_METRIC'] = 'curvef1'  # best.pt tracks curveIoU+F1, not the dead pixel-IoU

ROW = 'laneiou_both_full'               # SAME run name -- we are continuing it
PROJECT = '/content/runs/laneiou_full'  # SAME project
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)
EPOCHS, BATCH, LR0, PATIENCE = 120, 32, '4e-4', 30

# Pre-flight: the resume checkpoint must exist locally (restored in Cell 3c).
# --resume auto -> {PROJECT}/{ROW}/weights/last.pt.
_resume_ckpt = Path(PROJECT) / ROW / 'weights' / 'last.pt'
assert _resume_ckpt.exists(), (
    f'resume checkpoint missing: {_resume_ckpt}\\n'
    'Run Cell 3c ("Restore the NB107 checkpoint from Drive") first.')

cmd = [sys.executable, '-u', str(TRAIN),
       '--mode', 'full', '--model-yaml', str(MODEL), '--data-yaml', str(DATA),
       '--project', PROJECT, '--name', ROW,
       '--save-period', '10', '--batch', str(BATCH), '--epochs', str(EPOCHS),
       '--lr0', LR0, '--patience', str(PATIENCE),
       '--fliplr', '0.5', '--weight-decay', '0.05',
       '--lane-match', 'hungarian', '--lane-weights', 'clrkd', '--diff-clamp', '100',
       # === the NB98 winner combination (unchanged from NB107) ===
       '--use-aux-seg', '--aux-seg-classes', '2',
       '--aux-drivable-weight', '0.5', '--aux-lane-weight', '0.5',
       '--seg-gate-floor', '0.3',
       # === Sprint-2 winner: angle-aware LaneIoU in BOTH loss and matcher ===
       '--lane-iou-type', 'laneiou', '--lane-iou-match', 'laneiou',
       # === Sprint-0 free win: decode threshold ===
       '--lane-eval-tau', '0.4',
       # === NB108 deltas vs NB107 (attack the length-shrink) ==============
       '--resume', 'auto',              # continue from ep18 last.pt (-> 120)
       '--lane-y-reweight', 'near',     # S1.4 (INDIRECT): 1.85x near-row x-accuracy
       '--lane-len-hinge', '2.0']       # S1.4b (DIRECT): penalize too-short matched lanes
print('=== NB108 RESUME laneiou_both_full: last.pt + near y-reweight + len-hinge ===')
print('   ', ' '.join(cmd[3:]), flush=True)
rc = run_streaming(cmd, log_path=os.path.join(LOG_DIR, f'NB108_{ROW}.log'), check=False)
print(f'[{ROW}] rc={rc}  '
      '(CONFIRM the log showed "Resuming training ... from epoch ~19 to 120"; '
      'then watch length_mean recover + curveIoU/F1 hold/rise)')
'''


def build():
    nb = {
        "cells": [
            md(INTRO),
            md(nb107.CELL1_MD), code(nb107.CELL1),
            md(nb107.CELL2_MD), code(nb107.CELL2),
            md(nb107.CELL3_MD), code(nb107.CELL3),
            md(nb107.CELL_INSPECT_MD), code(nb107.CELL_INSPECT),
            md(CELL_RESTORE_MD), code(CELL_RESTORE),
            md(CELL4_MD), code(CELL4),
            md(nb107.CELL5_MD), code(nb107.CELL5),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_108_laneiou_resume_yreweight.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
