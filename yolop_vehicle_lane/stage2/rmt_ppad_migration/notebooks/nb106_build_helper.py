"""One-shot builder for stage2_notebook_106_sprint2_laneiou.ipynb.

NB106 - Sprint-2 S2.A: CLRerNet angle-aware LaneIoU. Sprint-1 (NB105) FAILED
gate G1: cheap cls/decode tweaks could not break the F1=0.148 plateau, and
Sprint-0 already showed the classifier ranks lanes well (AUC 0.96). So the
binding constraint is GEOMETRY -> the representation/loss, not calibration.

S2.A swaps the fixed-band LineIoU for CLRerNet's LaneIoU, whose virtual lane
half-width scales with the LOCAL SLOPE (lane_width * sqrt(dx^2+dy^2)/dy). That
widens the IoU band where the lane is steep -- the near-field, where our
jaggedness lives -- so shape error is penalized proportional to local angle.
LineIoU's fixed band under-penalizes exactly there. The port is unit-tested
against CLRerNet's own implementation (tests/test_lane_iou.py).

Grid (10k/15ep, tau=0.4 banked on every row like NB105):
  s2_baseline      line     - REUSED from NB105 round 1 (F1=0.148), not re-run
  s2_laneiou_loss  laneiou  - LaneIoU in the regression loss only
  s2_laneiou_both  laneiou  - LaneIoU in loss AND the hungarian matcher cost

Watch curveIoU (should rise past the ~0.09 saturation), lane_f1, and mAP50
(>=0.48 guardrail). Gate G2: F1 >= 0.45 / curveIoU >= 0.25 -> promote to 70k;
else -> the anchor paradigm is the wall -> Sprint 3 (curve-query DETR / MapTR).

Run this script ONCE to (re)generate the notebook; keep it in version control.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": [t]}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": [t]}


INTRO = """\
# NB106 - Sprint-2 S2.A: CLRerNet angle-aware LaneIoU

Sprint-1 (NB105) **failed G1**: F1 ceilinged at 0.148 and IoU-aware cls hurt
twice. With Sprint-0 showing the classifier already ranks well (AUC 0.96), the
wall is **geometry**. S2.A replaces the fixed-band LineIoU with CLRerNet's
**LaneIoU** - a slope-scaled IoU band that penalizes near-field shape error
proportional to local angle (our jaggedness zone). Port is unit-tested vs
CLRerNet's own code (`tests/test_lane_iou.py`: ours == ref to 2e-3).

| row | IoU type | where | tests |
|---|---|---|---|
| `s2_baseline`      | line    | -          | **REUSED from NB105 (F1=0.148)** - not re-run |
| `s2_laneiou_loss`  | laneiou | loss only  | does angle-aware loss lift curveIoU off ~0.09? |
| `s2_laneiou_both`  | laneiou | loss+match | + geometry-faithful assignment (full CLRerNet recipe) |

> Only **2 GPU rows** launch; the baseline is reused (loss byte-identical with
> `LANE_IOU_TYPE=line`). tau=0.4 banked on every row so F1 is comparable.

**Gate G2:** F1 >= ~0.45 AND curveIoU >= ~0.25 AND mAP50 >= 0.48 -> promote the
winner to 70k. If curveIoU still saturates ~0.09 -> the anchor/prior paradigm is
the limit -> Sprint 3 (curve-query DETR, MapTR template).
"""

CELL1_MD = "### Cell 1: Mount + deps + locate (mount-alive guarded)"
CELL1 = '''\
import os, sys, subprocess
from pathlib import Path
os.environ['PYTHONIOENCODING'] = 'utf-8'
REPO_ROOT='/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
MIG=Path(REPO_ROOT)/'stage2/rmt_ppad_migration'
TRAIN=MIG/'P8_train/scripts/train_lane_only.py'
MODEL=MIG/'vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml'
DATA =MIG/'vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml'
DATASETS=Path('/content/drive/MyDrive/EcoCAR/datasets')
SUBSET=Path('/content/bdd_subset_10k')
def _alive():
    try: return os.path.isdir('/content/drive/MyDrive')
    except OSError: return False
if not _alive():
    try:
        from google.colab import drive; drive.mount('/content/drive', force_remount=True)
    except Exception as e: print('[mount]', e)
if not _alive():
    raise RuntimeError('Drive mount DEAD (OSError 107). Runtime -> Disconnect '
                       'and delete runtime, reconnect, re-run from Cell 1.')
os.chdir(REPO_ROOT); sys.path.insert(0, REPO_ROOT)
for _p in ('addict','yapf','scipy'):
    try: __import__(_p)
    except ImportError: subprocess.check_call([sys.executable,'-m','pip','install','-q',_p])
try: import mmcv
except ImportError: subprocess.check_call([sys.executable,'-m','pip','install','-q','mmcv'])
for p in (TRAIN,MODEL,DATA): assert p.exists(), f'missing {p}'
print('[ok] env ready')
'''

CELL2_MD = "### Cell 2: Extract the 10k subset + point the data YAML at it (idempotent)"
CELL2 = '''\
PREP=MIG/'extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py'
req=[DATASETS/'bdd100k_clrkd_curve.tar',
     Path('/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/BDD_detection_labels.zip'),
     DATASETS/'lane_targets_clr_v1_polyline.tar.gz']
miss=[str(p) for p in req if not p.exists()]
if miss: raise FileNotFoundError('Missing:\\n  '+'\\n  '.join(miss))
if (SUBSET/'prep_summary.json').exists() or (
        (SUBSET/'images/val2017').exists() and any((SUBSET/'images/val2017').iterdir())):
    print('[ok] subset present; skipping')
else:
    cmd=[sys.executable,'-u',str(PREP),'--out-root',str(SUBSET),
         '--n-train','10000','--n-val','2000','--seed','89']
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout: print(line,end='',flush=True)
    if p.wait()!=0: raise RuntimeError('subset prep failed')
def _yset(y,f,v):
    txt=y.read_text(encoding='utf-8'); line=f'{f}: {v}'
    if line in txt: return
    L=txt.splitlines()
    for i,ln in enumerate(L):
        if ln.strip().startswith(f'{f}:'): L[i]=line; break
    else: L.append(line)
    y.write_text('\\n'.join(L)+'\\n',encoding='utf-8'); print(f'  [yaml] {f} -> {v}')
def _yunset(y,f):
    txt=y.read_text(encoding='utf-8')
    keep=[ln for ln in txt.splitlines() if not ln.strip().startswith(f'{f}:')]
    n='\\n'.join(keep)+'\\n'
    if n!=txt: y.write_text(n,encoding='utf-8'); print(f'  [yaml] removed {f}')
_yset(DATA,'path','/content/bdd_subset_10k')
_yset(DATA,'lane_targets_root','/content/bdd_subset_10k/lane_targets')
_yunset(DATA,'drivable_masks_root')
print('[ok] data yaml configured')
'''

CELL3_MD = """\
### Cell 3: `launch(name, iou_type=...)` helper

NB101 recipe (clrkd, hungarian, clamp=100, fliplr=0.5, wd=0.05) + tau=0.4
banked. Only `--lane-iou-type` varies. Resume-safe; syncs to Drive each epoch.
"""
CELL3 = '''\
from stage2.scripts.notebook_utils import run_streaming
PROJECT='/content/runs/sprint2'
LOG_DIR='/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'; os.makedirs(LOG_DIR, exist_ok=True)
EPOCHS,BATCH,LR0,PATIENCE=15,32,'4e-4',99
def launch(name, iou_type='line', iou_match='follow'):
    # iou_type = LaneIoU in the LOSS; iou_match = LaneIoU in the MATCHER
    # ('follow' = same as iou_type). loss-only -> iou_type='laneiou',
    # iou_match='line'; both -> iou_type='laneiou', iou_match='laneiou'.
    #
    # FULL NB101-WINNER RECIPE: every row inherits the proven MTL engineering
    # (aux dense drivable+lane seg + GCA lane gate floor 0.3) on top of the
    # NB97 polyline recipe, so S2.A LaneIoU is tested on the SAME base that
    # gave us combined_aux_gca_full (F1=0.30, the project's best). Without
    # these four flags, NB106 would be testing LaneIoU on a degraded base and
    # any null result would be ambiguous.
    cmd=[sys.executable,'-u',str(TRAIN),
         '--mode','full','--model-yaml',str(MODEL),'--data-yaml',str(DATA),
         '--project',PROJECT,'--name',name,'--device','0',
         '--save-period','15','--batch',str(BATCH),'--epochs',str(EPOCHS),
         '--lr0',LR0,'--patience',str(PATIENCE),
         # NB97 polyline recipe (carry-over):
         '--fliplr','0.5','--weight-decay','0.05',
         '--lane-match','hungarian','--lane-weights','clrkd','--diff-clamp','100',
         # NB101 winner MTL flags (aux_seg + gca_floor) -- mandatory:
         '--use-aux-seg','--aux-seg-classes','2',
         '--aux-drivable-weight','0.5','--aux-lane-weight','0.5',
         '--seg-gate-floor','0.3',
         # Sprint-0 / Sprint-2 levers under test:
         '--lane-eval-tau','0.4',
         '--lane-iou-type',iou_type,'--lane-iou-match',iou_match]
    print(f'\\n=== {name}: loss-iou={iou_type} match-iou={iou_match} '
          f'(NB101 winner recipe: aux_seg + gca_floor, tau=0.4) ===\\n', flush=True)
    rc=run_streaming(cmd, log_path=os.path.join(LOG_DIR,f'NB106_{name}.log'), check=False)
    print(f'[{name}] rc={rc}  (watch curveIoU, lane_f1, mAP50)')
    return rc==0
print('[ready]')
'''

ROWS = [
    ('s2_baseline', "Row 0: baseline - REUSE NB105 round-1 weights (do NOT re-run)",
     "# Baseline REUSED from NB105 (LANE_IOU_TYPE=line -> loss byte-identical).\n"
     "# F1=0.148, curveIoU=0.517, mAP50=0.671 @15ep. Aggregator reads it.\n"
     "print('[baseline] reusing NB105 s1_baseline (F1=0.148) - not re-run')"),
    ('s2_laneiou_loss', "Row 1: LaneIoU in the regression LOSS ONLY (matcher stays LineIoU)",
     "launch('s2_laneiou_loss', iou_type='laneiou', iou_match='line')"),
    ('s2_laneiou_both', "Row 2: LaneIoU in loss AND matcher cost (full CLRerNet recipe)",
     "launch('s2_laneiou_both', iou_type='laneiou', iou_match='laneiou')"),
]

CELL_AGG_MD = """\
### Cell 9: Aggregate - rank by curveIoU + lane_f1, check G2

curveIoU is the key Sprint-2 signal (it was saturated ~0.09 -> LaneIoU should
lift it). G2 = F1>=0.45 AND curveIoU>=0.25 AND mAP50>=0.48.
"""
CELL_AGG = '''\
import csv
from pathlib import Path
PROJECT=Path('/content/runs/sprint2')
DRIVE_CK=Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints')
ROWS=['s2_baseline','s2_laneiou_loss','s2_laneiou_both']
_BASE={'metrics/lane_f1(lane)':'0.1483','metrics/lane_curveIoU(lane)':'0.5166',
       'metrics/lane_score_std(lane)':'0.0824','metrics/lane_length_mean(lane)':'0.1052',
       'metrics/mAP50(B)':'0.6712'}
def final(name):
    # s2_baseline reuses NB105's s1_baseline folder.
    cands=[PROJECT/name, DRIVE_CK/name]
    if name=='s2_baseline': cands += [Path('/content/runs/sprint1/s1_baseline'), DRIVE_CK/'s1_baseline']
    for b in cands:
        p=b/'results.csv'
        if p.exists():
            rows=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(open(p))]
            if rows: return rows[-1]
    if name=='s2_baseline': return _BASE
    return None
def g(r,*ks):
    for k in ks:
        for ck in (r or {}):
            if ck.strip().lower()==k.lower():
                try: return float(r[ck])
                except: pass
    return None
print(f'{"row":18} {"F1":>7} {"curveIoU":>9} {"mAP50":>7} {"len_mean":>9} {"G2":>4}')
print('-'*60)
best=None
for n in ROWS:
    r=final(n)
    if r is None: print(f'{n:18} (not run)'); continue
    f1=g(r,'metrics/lane_f1(lane)'); ci=g(r,'metrics/lane_curveIoU(lane)')
    mp=g(r,'metrics/mAP50(B)'); lm=g(r,'metrics/lane_length_mean(lane)')
    g2=(f1 or 0)>=0.45 and (ci or 0)>=0.25 and (mp or 0)>=0.48
    def fmt(x): return f'{x:.4f}' if isinstance(x,float) else '  -  '
    print(f'{n:18} {fmt(f1):>7} {fmt(ci):>9} {fmt(mp):>7} {fmt(lm):>9} {("Y" if g2 else "-"):>4}')
    if f1 is not None and (mp or 0)>=0.48 and (best is None or f1>best[1]): best=(n,f1,ci)
print()
if best: print(f'BEST (mAP50>=0.48): {best[0]}  F1={best[1]:.4f}  curveIoU={best[2]:.4f}')
print('G2: F1>=0.45 + curveIoU>=0.25 + mAP50>=0.48 -> promote to 70k.')
print('curveIoU rising off ~0.09 = LaneIoU working even if F1 short of 0.45.')
print('If curveIoU still ~0.09 -> anchor paradigm is the wall -> Sprint 3 (curve-query DETR).')
'''


def build():
    cells=[md(INTRO), md(CELL1_MD),code(CELL1), md(CELL2_MD),code(CELL2), md(CELL3_MD),code(CELL3)]
    for i,(name,title,call) in enumerate(ROWS, start=4):
        cells.append(md(f"### Cell {i}: {title}")); cells.append(code(call+"\n"))
    cells.append(md(CELL_AGG_MD)); cells.append(code(CELL_AGG))
    nb={"cells":cells,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.11"}},
        "nbformat":4,"nbformat_minor":5}
    out=Path(__file__).resolve().parent/'stage2_notebook_106_sprint2_laneiou.ipynb'
    out.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
