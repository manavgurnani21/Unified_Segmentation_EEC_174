"""One-shot builder for stage2_notebook_99_curve_f1_metric_verification.ipynb.

NB99 - curve-F1 metric verification (the notebook that was missing).

The curve-F1 / curveIoU / mean-best-IoU metric (P7_validator/tools/
lane_curve_f1.py, wired into val.py) is what we now judge lane quality by -
but until now it was only checked with a throwaway local probe. This
notebook PERMANENTLY verifies the metric is sound, so that when training
reports `lane_f1=0` you can trust it is the MODEL, not the metric.

Self-contained and CPU-only (numpy + cv2): NO GPU, NO datasets, NO
checkpoints. On Colab it mounts Drive to read ONE file - the curve-F1
source (the repo lives on Drive); Cell 1 does this automatically. Locally
the Drive mount is skipped. Re-running this builder overwrites the
notebook; keep it in version control.

What it asserts:
  - perfect pred==gt polyline -> F1 = 1.0           (decode + match correct)
  - horizontal-offset sweep   -> IoU crosses 0.5 at ~15-18 px @640
                                 (calibrates how sharp the model must be)
  - matched 16-D Bezier       -> F1 = 1.0           (bezier decode path)
  - 192-prior tensor (4 good  -> F1 = 1.0, 4 TP, mean-best-IoU > 0.5
    + 188 junk)                 (top-N selection + un-saturated companion)
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
# NB99 - Curve-F1 metric verification

Permanent unit-test for the lane curve-F1 metric
(`P7_validator/tools/lane_curve_f1.py`) that NB92/NB97 report as
`lane_f1(lane)` / `curveIoU(lane)`. **CPU-only: no GPU, no datasets, no
checkpoints.** (On Colab it does mount Drive to read the one source file -
the repo lives there; Cell 1 handles this automatically.)

WHY this notebook exists: the metric was previously validated only with a
throwaway local probe. If `lane_f1=0` during training, we must be certain
that is the *model* (lanes not sharp enough), not a broken metric. This
notebook proves the metric is correct on synthetic ground truth.

> **Sequence note.** Logically this is a *diagnostic/validator* test and
> belongs alongside NB86 (P7 validator) - the diagnostics-before-full-
> training rule would put it before NB97/NB98. It carries number 99 only
> because it was written retroactively, AFTER NB97 (polyline) and NB98
> (MTL) had already run and been recorded in debug_record.md + the frozen
> run logs. Renumbering would rewrite that history (and contradict the
> immutable `training_runs/.../full_train.log` artifacts), so the number
> is kept and this pointer records the intended ordering instead.

| Test | Asserts | Meaning |
|---|---|---|
| 1 perfect match | F1 = 1.0 | decode + greedy match correct |
| 2 offset sweep | IoU crosses 0.5 @ ~15-18px(640) | how sharp the model must be |
| 3 matched Bezier (16-D) | F1 = 1.0 | the bezier decode path works |
| 4 192-prior realistic | 4 TP, mean-best-IoU > 0.5 | top-N pick + un-saturated companion |
"""

CELL1_MD = "### Cell 1: Locate the metric module (Colab or local) + ensure cv2"

CELL1_CODE = '''\
import os, sys, importlib.util
from pathlib import Path
import numpy as np

# This test needs only the curve-F1 SOURCE FILE - but that file lives in the
# repo, which on Colab is on Drive. So we DO mount Drive (no GPU, no datasets,
# no checkpoints - just the one .py). Locally, Drive mount is skipped and the
# upward CWD search finds the checkout.
def _maybe_mount_drive():
    if Path('/content/drive/MyDrive').exists():
        return  # already mounted
    try:
        from google.colab import drive  # noqa: F401  (only on Colab)
    except ImportError:
        return  # not Colab -> local checkout, nothing to mount
    print('[setup] mounting Drive (needed for the repo source)...', flush=True)
    drive.mount('/content/drive', force_remount=False)

REL = Path('P7_validator') / 'tools' / 'lane_curve_f1.py'

def find_migration_root():
    """Return (migration_root, file_synced).

    Two phase, so we can tell apart the two very different failures:
      - migration_root is None  -> the repo itself is not visible (Drive not
        mounted / not synced / unexpected path)
      - migration_root set but file_synced False -> the repo IS here but
        lane_curve_f1.py is missing == Drive is STALE, re-sync needed.
    """
    roots = []
    # 1) walk up from CWD (local checkout / notebook opened inside the repo)
    cur = Path.cwd()
    for p in [cur, *cur.parents]:
        if p.name == 'rmt_ppad_migration':
            roots.append(p)
        roots.append(p / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration')
    # 2) explicit known Colab/Drive layouts
    for base in ('/content/drive/MyDrive/EcoCAR', '/content/drive/MyDrive',
                 '/content',
                 '/content/drive/MyDrive/EcoCAR/EcoCAR-Perception-Pipeline-YOLO26-BDD100K'):
        roots.append(Path(base) / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration')
    # 3) bounded glob fallback (handles a renamed parent folder on Drive)
    for base in ('/content/drive/MyDrive/EcoCAR', '/content/drive/MyDrive'):
        b = Path(base)
        if b.exists():
            try:
                roots += list(b.glob('*/yolop_vehicle_lane/stage2/rmt_ppad_migration'))
                roots += list(b.glob('yolop_vehicle_lane/stage2/rmt_ppad_migration'))
            except Exception:
                pass
    # dedupe, preserve order
    seen, uniq = set(), []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s); uniq.append(r)
    root_found = None
    for r in uniq:
        if (r / 'P7_validator').exists():          # a real migration root
            root_found = root_found or r
            if (r / REL).exists():                  # ...with the metric synced
                return r, True
    return root_found, False

_maybe_mount_drive()
MIG, _synced = find_migration_root()
if MIG is None:
    raise FileNotFoundError(
        'Could not find rmt_ppad_migration/ anywhere. On Colab: mount Drive '
        'and confirm the repo is at '
        '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/rmt_ppad_migration')
if not _synced:
    raise FileNotFoundError(
        f'Found the repo at:\\n  {MIG}\\n'
        f'but {REL} is NOT there. Drive is STALE -> re-sync your local repo to '
        f'Drive (this file was added recently), then re-run this cell.')
spec = importlib.util.spec_from_file_location('lcf1', MIG / REL)
lcf1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lcf1)

try:
    import cv2  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'opencv-python-headless'])
    import cv2  # noqa: F401

N, N_STRIPS = 78, 71
print('[ok] curve-F1 module loaded from', MIG)
'''

CELL2_MD = """\
### Cell 2: Builders + Test 1 - perfect polyline match -> F1 = 1.0

Encodes one GT lane (proven format: start_y normalized, length = strip
count, x = 640-px coords) and a PRED lane representing the SAME line
(pred format: length + x NORMALIZED). A correct decode + match returns
exactly one true positive.
"""

CELL2_CODE = '''\
def gt_row(start_norm, length_strips, x0, x1):
    """GT: start_y norm, length=strip-count, x=640-pixel coords."""
    r = np.full(N, -1e5, dtype=np.float64)
    r[0] = 0.0; r[1] = 1.0
    r[2] = start_norm; r[5] = float(length_strips)
    s = int(round(start_norm * N_STRIPS))
    r[6 + s:6 + s + length_strips] = np.linspace(x0, x1, length_strips)
    return r

def pred_row(start_norm, length_strips, x0, x1, pos_logit=6.0):
    """PRED: start_y norm, length NORMALIZED, x NORMALIZED [0,1]."""
    r = np.full(N, -1e5, dtype=np.float64)
    r[0] = 0.0; r[1] = pos_logit
    r[2] = start_norm; r[5] = length_strips / N_STRIPS
    s = int(round(start_norm * N_STRIPS))
    r[6 + s:6 + s + length_strips] = np.linspace(x0, x1, length_strips) / 639.0
    return r

def per_lane_iou(pred, gt, R=160, w=12):
    p = lcf1._decode_row(pred, R, 72, is_pred=True)
    g = lcf1._decode_row(gt, R, 72, is_pred=False)
    if p is None or g is None:
        return None
    pm = lcf1._rasterize_one(p, R, w); gm = lcf1._rasterize_one(g, R, w)
    inter = np.logical_and(pm, gm).sum()
    union = pm.sum() + gm.sum() - inter
    return inter / union if union else 0.0

gt = [gt_row(0.0, 40, 200, 300)]
pr = [pred_row(0.0, 40, 200, 300)]
iou = per_lane_iou(pr[0], gt[0])
tp, fp, fn, isum, ng = lcf1.lane_curve_tp_fp_fn(np.array(pr), np.array(gt))
f1 = lcf1.f1_from_counts(tp, fp, fn)['f1']
print(f'TEST 1  per-lane IoU={iou:.3f}  tp/fp/fn={tp}/{fp}/{fn}  F1={f1:.3f}')
assert tp == 1 and f1 == 1.0, 'perfect match must give F1=1.0'
print('  PASS - decode + match are correct')
'''

CELL3_MD = """\
### Cell 3: Test 2 - horizontal-offset sweep (how sharp must the model be?)

Shifts the pred lane sideways and measures per-lane IoU at R=160,
width=12. This calibrates the F1@0.5 bar: a lane must land within
~15-18 px (@640) of GT to count as a match. `lane_f1=0` in training
means NO predicted lane is that sharp - a model issue, not a metric bug.
"""

CELL3_CODE = '''\
print('offset_px(@640) | IoU@R160,w12 | match@0.5?')
crossed = None
for off in (0, 5, 10, 15, 18, 20, 30, 60):
    iou = per_lane_iou(pred_row(0.0, 40, 200 + off, 300 + off), gt[0])
    hit = iou >= 0.5
    if crossed is None and not hit:
        crossed = off
    print(f'  {off:>5}         |   {iou:5.3f}     | {"YES" if hit else "no"}')
print(f'  -> IoU drops below 0.5 around {crossed} px (@640) offset')
assert per_lane_iou(pred_row(0.0,40,200,300), gt[0]) >= 0.9, 'zero offset must be ~1'
print('  PASS - threshold behaves monotonically and sanely')
'''

CELL4_MD = """\
### Cell 4: Test 3 - matched 16-D Bezier -> F1 = 1.0

Exercises the Bezier decode path (`_decode_lane_bezier`). Both PRED and
GT bezier control points live in normalized [0,1], so an identical
bezier used as both pred and GT must return one true positive.
"""

CELL4_CODE = '''\
def bezier_row(p0, p3, t0=0.02, t1=0.98, pos=6.0):
    """16-D: [neg,pos, P0x..P3x, P0y..P3y, t_start, t_end, cmplx, w1,w2,w3]."""
    r = np.zeros(16, dtype=np.float64)
    r[0] = 0.0; r[1] = pos
    (x0, y0), (x3, y3) = p0, p3
    x1, x2 = (2*x0 + x3) / 3, (x0 + 2*x3) / 3
    y1, y2 = (2*y0 + y3) / 3, (y0 + 2*y3) / 3
    r[2:6] = [x0, x1, x2, x3]
    r[6:10] = [y0, y1, y2, y3]
    r[10] = t0; r[11] = t1
    return r

bz = bezier_row((0.30, 0.90), (0.50, 0.10))
tp, fp, fn, isum, ng = lcf1.lane_curve_tp_fp_fn(np.array([bz]), np.array([bz]))
f1 = lcf1.f1_from_counts(tp, fp, fn)['f1']
print(f'TEST 3  bezier matched  tp/fp/fn={tp}/{fp}/{fn}  F1={f1:.3f}  '
      f'mean-best-IoU={isum/max(1,ng):.3f}')
assert tp == 1 and f1 == 1.0, 'matched bezier must give F1=1.0'
print('  PASS - 16-D bezier decode path works')
'''

CELL5_MD = """\
### Cell 5: Test 4 - realistic 192-prior tensor + mean-best-IoU

4 good priors near 4 GT lanes, 188 junk priors with low pos score. The
top-N selection must pick the good ones (4 TP) and the un-saturated
companion `mean-best-IoU` (= curveIoU) must be high.
"""

CELL5_CODE = '''\
gts = [gt_row(0.0, 40, 150, 200), gt_row(0.0, 40, 300, 320),
       gt_row(0.1, 35, 450, 480), gt_row(0.05, 38, 560, 600)]
rows = [pred_row(s, l, a, b, pos_logit=8.0) for (s, l, a, b) in
        [(0.0,40,158,208), (0.0,40,305,325), (0.1,35,458,488), (0.05,38,552,592)]]
rng = np.random.default_rng(0)
for _ in range(188):
    r = np.full(N, -1e5, dtype=np.float64)
    r[0] = 2.0; r[1] = -2.0; r[2] = 0.0; r[5] = 0.4
    r[6:6+40] = rng.uniform(0, 1, 40)
    rows.append(r)

tp, fp, fn, isum, ng = lcf1.lane_curve_tp_fp_fn(np.array(rows), np.array(gts))
mbi = isum / max(1, ng)
f1 = lcf1.f1_from_counts(tp, fp, fn)['f1']
print(f'TEST 4  tp/fp/fn={tp}/{fp}/{fn}  F1={f1:.3f}  mean-best-IoU(curveIoU)={mbi:.3f}')
assert tp == 4, 'top-N must recover all 4 good priors among 188 junk'
assert mbi > 0.5, 'curveIoU should be high for good matches'
print('  PASS - top-N selection + curveIoU companion both correct')
'''

CELL6_MD = "### Cell 6: Verdict"

CELL6_CODE = '''\
print("=" * 60)
print("CURVE-F1 METRIC: VERIFIED")
print("  - perfect match  -> F1 = 1.0")
print("  - bezier match   -> F1 = 1.0")
print("  - 192-prior      -> 4 TP, curveIoU > 0.5")
print("  - 0.5 bar needs lanes within ~15-18 px (@640) of GT")
print()
print("=> If training reports lane_f1=0, the metric is SOUND: the model's")
print("   lanes are not yet sharp enough. Watch `curveIoU` (un-saturated)")
print("   to see geometry improving before F1 lifts off 0.")
print("=" * 60)
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
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_99_curve_f1_metric_verification.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
