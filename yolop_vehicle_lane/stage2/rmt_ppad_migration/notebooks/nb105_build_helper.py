"""One-shot builder for stage2_notebook_105_sprint1_ablation.ipynb.

NB105 - Sprint-1 lane-head ablation grid (NEXT_STEPS sec 5), DATA-CORRECTED by
the NB104 Sprint-0 findings:

  S0.2 cls AUC = 0.96  -> the classifier RANKS lanes well; a threshold WORKS.
  S0.3 best decode-F1 = 0.346 @ tau=0.4 (vs 0.305) AND cnt_corr 0->0.64
       -> S1.1 thresholded decode is a FREE +13% F1 + the "always-8" cardinality
          fix. We BANK it on every row (--lane-eval-tau 0.4), no retrain needed.
  S0.4 jaggedness near/far = 0.49x  -> near is SMOOTHER than far (opposite of the
       NB103 visual). So S1.2 smoothness-reg loses its near-field rationale and is
       demoted to ONE optional row, re-judged after thresholding.

Grid (10k subset, ~15-ep probes; every row evals at tau=0.4 so F1 is comparable
to the thresholded decode):

  row              S1.1 tau   S1.3 iou-cls   S1.4 y-reweight   S1.2 smooth
  baseline_tau       0.4         off            none             0
  iou_cls            0.4         2.0            none             0     <- lift ceiling
  yreweight          0.4         off            near             0     <- stop length-shrink
  iou_cls+yrw        0.4         2.0            near             0     <- the likely winner
  +smooth (opt)      0.4         2.0            near             0.05  <- re-check S1.2

Watch: lane_f1 (>=0.32 = G1), lane_score_std (S1.3 should push it past 0.09),
lane_length_mean (S1.4 should stop the 0.38->0.28 shrink), mAP50 (>=0.48
guardrail). Promote the winner to 70k (NB101 recipe) only if G1 is met.

Run this script ONCE to (re)generate the notebook; keep it in version control.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": [t]}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": [t]}


INTRO = """\
# NB105 - Sprint-1 lane-head ablation (data-corrected by NB104)

Small, composable changes to the NB101 polyline head, ablated on the 10k subset
(~15-ep probes). **S1.1 confidence threshold (tau=0.4) is BANKED on every row**
(NB104 S0.3: free +13% F1 + fixes "always 8 lanes"); the rows test the training
levers that still have a rationale after Sprint 0.

**Round 2 (lean re-run).** Round 1 hit a bug: S1.3's QFL used the *raw* matched
line-IoU (~0.05 early) as the target, collapsing score_std 0.082->0.005 -> tau
rejected all -> F1=0. FIXED (soft target = floor 0.5 + quality; regression-
tested). Round 1 also showed y-reweight 'near' HURT F1 (abandoned the scoring
far-field). So round 2 drops the broken combos and the aggressive 'near':

| row | S1.3 IoU-cls (fixed) | S1.4 y-reweight | tests |
|---|---|---|---|
| `s1_baseline`      | off | none  | **REUSED from round 1** (F1=0.148) - not re-run (loss byte-identical with levers off) |
| `s1b_iou_cls`      | 2.0 | none  | does the FIXED QFL spread score_std + lift F1? |
| `s1b_iou_mildyrw`  | 2.0 | angle | + mild ('angle' 0.45/0.55) reweight, not the hurtful 'near' |

> Only 2 GPU rows this round - the baseline is reused, so you launch
> `s1b_iou_cls` and `s1b_iou_mildyrw` only.

**Decision gate G1:** if a row reaches **F1 >= ~0.32, score_std past 0.09,
length_mean stops shrinking, mAP50 >= 0.48** -> promote it to 70k. If F1 still
ceilings < 0.30 despite threshold-able cls -> representation is the wall ->
Sprint 2 (CLRerNet LaneIoU).

> Levers are env-driven in MTDETRDLoss (`LANE_IOU_CLS`, `LANE_Y_REWEIGHT`,
> `LANE_SMOOTH_W`, `LANE_EVAL_TAU`), set by train_lane_only's new CLI flags. All
> default OFF => byte-identical to NB101 when unset. Helper math is unit-tested
> (tests/test_sprint1_levers.py).
"""

CELL1_MD = "### Cell 1: Mount + deps + locate (mount-alive guarded)"
CELL1 = '''\
import os, sys, subprocess
from pathlib import Path
os.environ['PYTHONIOENCODING'] = 'utf-8'

REPO_ROOT  = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
MIG        = Path(REPO_ROOT)/'stage2/rmt_ppad_migration'
TRAIN      = MIG/'P8_train/scripts/train_lane_only.py'
MODEL      = MIG/'vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml'
DATA       = MIG/'vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml'
DATASETS   = Path('/content/drive/MyDrive/EcoCAR/datasets')
SUBSET     = Path('/content/bdd_subset_10k')

def _alive():
    try: return os.path.isdir('/content/drive/MyDrive')
    except OSError: return False
if not _alive():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
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
for p in (TRAIN, MODEL, DATA):
    assert p.exists(), f'missing {p} (sync Drive)'
print('[ok] env ready')
'''

CELL2_MD = """\
### Cell 2: Extract the 10k subset + point the data YAML at it
Idempotent. Same subset NB98/NB92 used (lane-only; drivable removed for the
clean collate path).
"""
CELL2 = '''\
PREP = MIG/'extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py'
req = [DATASETS/'bdd100k_clrkd_curve.tar',
       Path('/content/drive/MyDrive/EcoCAR/downloads/rmt_ppad_weights/BDD_detection_labels.zip'),
       DATASETS/'lane_targets_clr_v1_polyline.tar.gz']
missing=[str(p) for p in req if not p.exists()]
if missing: raise FileNotFoundError('Missing:\\n  '+'\\n  '.join(missing))
if (SUBSET/'prep_summary.json').exists() or (
        (SUBSET/'images/val2017').exists() and any((SUBSET/'images/val2017').iterdir())):
    print('[ok] subset present; skipping')
else:
    cmd=[sys.executable,'-u',str(PREP),'--out-root',str(SUBSET),
         '--n-train','10000','--n-val','2000','--seed','89']
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout: print(line,end='',flush=True)
    if p.wait()!=0: raise RuntimeError('subset prep failed')

def _yset(y, f, v):
    txt=y.read_text(encoding='utf-8'); line=f'{f}: {v}'
    if line in txt: return
    L=txt.splitlines()
    for i,ln in enumerate(L):
        if ln.strip().startswith(f'{f}:'): L[i]=line; break
    else: L.append(line)
    y.write_text('\\n'.join(L)+'\\n',encoding='utf-8'); print(f'  [yaml] {f} -> {v}')
def _yunset(y, f):
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
### Cell 3: `launch(name, **levers)` helper

Every row uses the NB101 recipe (clrkd, hungarian, clamp=100, fliplr=0.5,
wd=0.05) + **tau=0.4 eval threshold banked**, varying only the S1 training
levers. Resume-safe; best.pt/last.pt/full_train.log sync to Drive each epoch.
"""
CELL3 = '''\
from stage2.scripts.notebook_utils import run_streaming
PROJECT = '/content/runs/sprint1'
LOG_DIR = '/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs'
os.makedirs(LOG_DIR, exist_ok=True)
EPOCHS, BATCH, LR0, PATIENCE = 15, 32, '4e-4', 99   # 15-ep probe, no early-stop

def launch(name, iou_cls='off', y_reweight='none', smooth_w=0.0, tau=0.4):
    cmd=[sys.executable,'-u',str(TRAIN),
         '--mode','full','--model-yaml',str(MODEL),'--data-yaml',str(DATA),
         '--project',PROJECT,'--name',name,'--device','0',
         '--save-period','15','--batch',str(BATCH),'--epochs',str(EPOCHS),
         '--lr0',LR0,'--patience',str(PATIENCE),
         '--fliplr','0.5','--weight-decay','0.05',
         '--lane-match','hungarian','--lane-weights','clrkd','--diff-clamp','100',
         # S1 levers under test:
         '--lane-iou-cls',str(iou_cls),'--lane-y-reweight',str(y_reweight),
         '--lane-smooth-w',str(smooth_w),'--lane-eval-tau',str(tau)]
    print(f'\\n=== {name}: iou_cls={iou_cls} y_reweight={y_reweight} '
          f'smooth_w={smooth_w} tau={tau} ===\\n', flush=True)
    rc=run_streaming(cmd, log_path=os.path.join(LOG_DIR,f'NB105_{name}.log'), check=False)
    print(f'[{name}] rc={rc}  (watch lane_f1, lane_score_std, length_mean, mAP50)')
    return rc==0
print('[ready] call launch(...) per row cell below')
'''

# LEAN RE-RUN (NB105 round 2). Round 1 found: (a) S1.3 QFL used the raw matched
# IoU (~0.05 early) as the target -> drove all positive scores to ~0.05 ->
# score_std collapsed 0.082->0.005 -> tau=0.4 rejected every lane -> F1=0. FIXED
# in loss.py (soft target now floor 0.5 + quality, regression-tested). (b)
# y-reweight 'near' (0.35/0.65) HURT F1 0.148->0.043 by abandoning the scoring
# far-field. So this round: baseline bar, the FIXED iou_cls alone, and iou_cls +
# the MILDER 'angle' reweight (0.45/0.55) -- drops the disproven aggressive
# 'near' and the broken combos.
ROWS = [
    ('s1_baseline', "Row 0: baseline - REUSE round-1 weights (do NOT re-run)",
     "# Baseline is REUSED from round 1, not re-launched. All my loss edits are\n"
     "# gated behind iou_cls/y_reweight/smooth flags; with them OFF the loss is\n"
     "# byte-identical to round 1, so s1_baseline's result (F1=0.148, score_std\n"
     "# =0.082, mAP50=0.671 @15ep) is unchanged. The aggregator reads the\n"
     "# existing /content/runs/sprint1/s1_baseline/results.csv (or the recorded\n"
     "# numbers as a fallback). Nothing to run here.\n"
     "print('[baseline] reusing round-1 s1_baseline (F1=0.148) - not re-run')"),
    ('s1b_iou_cls',   "Row 1: FIXED S1.3 IoU-aware cls alone (QFL gamma=2.0, floor=0.5)",
     "launch('s1b_iou_cls', iou_cls='2.0')"),
    ('s1b_iou_mildyrw', "Row 2: FIXED iou_cls + MILD y-reweight ('angle', not 'near')",
     "launch('s1b_iou_mildyrw', iou_cls='2.0', y_reweight='angle')"),
]

CELL_AGG_MD = """\
### Cell 9: Aggregate - rank rows by lane_f1 (@tau=0.4) + check the guardrail

Reads each row's results.csv final epoch. WINNER = highest lane_f1 with
mAP50 >= 0.48 AND score_std visibly > 0.09 (cls now threshold-able) AND
length_mean not shrinking. That row is promoted to 70k.
"""
CELL_AGG = '''\
import csv
from pathlib import Path
PROJECT = Path('/content/runs/sprint1')
DRIVE_CK = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints')
ROWS = ['s1_baseline','s1b_iou_cls','s1b_iou_mildyrw']
# Baseline reused from round 1: its result is unchanged (loss byte-identical
# with levers off). Hard-coded fallback = the round-1 final-epoch numbers so the
# table is complete even if /content was wiped and the folder isn't re-synced.
_BASELINE_R1 = {'metrics/lane_f1(lane)': '0.1483',
                'metrics/lane_curveIoU(lane)': '0.5166',
                'metrics/lane_score_std(lane)': '0.0824',
                'metrics/lane_length_mean(lane)': '0.1052',
                'metrics/mAP50(B)': '0.6712'}
def final(name):
    # search /content run dir, then the Drive checkpoints mirror.
    for base in (PROJECT/name, DRIVE_CK/name):
        p=base/'results.csv'
        if p.exists():
            rows=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(open(p))]
            if rows: return rows[-1]
    if name=='s1_baseline':
        return _BASELINE_R1     # reuse round-1 numbers (loss unchanged)
    return None
def g(r,*ks):
    for k in ks:
        for ck in (r or {}):
            if ck.strip().lower()==k.lower():
                try: return float(r[ck])
                except: pass
    return None
print(f'{"row":15} {"F1":>7} {"curveIoU":>9} {"score_std":>10} {"len_mean":>9} {"mAP50":>7} {"verdict":>9}')
print('-'*72)
best=None
for n in ROWS:
    r=final(n)
    if r is None: print(f'{n:15} (not run)'); continue
    f1=g(r,'metrics/lane_f1(lane)'); ci=g(r,'metrics/lane_curveIoU(lane)')
    ss=g(r,'metrics/lane_score_std(lane)'); lm=g(r,'metrics/lane_length_mean(lane)')
    mp=g(r,'metrics/mAP50(B)')
    ok = (f1 or 0)>=0.32 and (mp or 0)>=0.48
    def fmt(x): return f'{x:.4f}' if isinstance(x,float) else '  -  '
    print(f'{n:15} {fmt(f1):>7} {fmt(ci):>9} {fmt(ss):>10} {fmt(lm):>9} {fmt(mp):>7} {("G1 PASS" if ok else "-"):>9}')
    if f1 is not None and (mp or 0)>=0.48 and (best is None or f1>best[1]): best=(n,f1)
print()
if best: print(f'WINNER (mAP50>=0.48): {best[0]}  lane_f1={best[1]:.4f}')
print('G1: F1>=0.32 + score_std>0.09 + length not shrinking + mAP50>=0.48 -> promote to 70k.')
print('If F1 ceilings <0.30 with threshold-able cls -> Sprint 2 (CLRerNet LaneIoU).')
'''


def build():
    cells=[md(INTRO), md(CELL1_MD),code(CELL1), md(CELL2_MD),code(CELL2),
           md(CELL3_MD),code(CELL3)]
    for i,(name,title,call) in enumerate(ROWS, start=4):
        cells.append(md(f"### Cell {i}: {title}"))
        cells.append(code(call+"\n"))
    cells.append(md(CELL_AGG_MD)); cells.append(code(CELL_AGG))
    nb={"cells":cells,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.11"}},
        "nbformat":4,"nbformat_minor":5}
    out=Path(__file__).resolve().parent/'stage2_notebook_105_sprint1_ablation.ipynb'
    out.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
