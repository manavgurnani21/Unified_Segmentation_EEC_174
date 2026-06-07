"""One-shot builder for stage2_notebook_109_demo_images_and_video.ipynb.

NB109 - DEMO / VERIFICATION of the NEWEST RMT-PPAD model (NB108's resumed run,
best.pt = epoch 37 of the `laneiou_both_full` run). It:
  1. renders >= 8 validation images with the model's actual predictions
     (detection boxes + decoded CLR polyline lanes) next to ground truth, and
     saves each annotated prediction as its own PNG + a combined grid, and
  2. runs the model frame-by-frame on a video and writes an annotated MP4,
     following the video approach of stage1/notebooks/07_a5000_video_profile.ipynb.

All demo outputs (images + video) are written into the run's checkpoint folder
on Drive (DEMO_DIR), so they sit next to the weights they came from.

Reuses NB103's proven helpers verbatim (load_model / infer / draw / lane decode
/ pickle-name shims) via import, so the inference path is identical to the one
the curve-F1 metric scored.

Run this script ONCE to (re)generate the notebook; keep it in version control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb103_build_helper as nb103  # noqa: E402  reuse its data + helper cells verbatim

md, code = nb103.md, nb103.code


INTRO = """\
# NB109 - Demo & verification of the newest RMT-PPAD model (8+ images + video)

Qualitative verification of the **newest** RMT-PPAD lane+detection model: the
`laneiou_both_full` run resumed in NB108, whose **best.pt is epoch 37** (curve-F1
0.621, curveIoU 0.752, detection mAP50 0.835; the NB108 length fix recovered the
mean lane length 0.094 -> 0.103). The aux drivable/lane-seg heads are
training-only and dropped at eval, so what you see is the true deployable model
(detection boxes + CLR polyline lanes).

**What it produces:**
1. **>= 8 validation images** - ground truth vs. best.pt prediction, saved as
   individual annotated PNGs (`rmt_ppad_pred_<stem>.png`) plus a combined grid,
   for dropping straight into the report's RMT-PPAD demonstration figure.
2. **An annotated video** - the model run frame-by-frame over a clip, written as
   an MP4 with a measured FPS overlay (same video approach as Stage-1 NB07).

**Where outputs go:** the run's checkpoint folder on Drive,
`.../training_runs/checkpoints/laneiou_both_full/demo/` (set by `DEMO_DIR`).

> If your newest `best.pt` lives under a different run folder, change `RUN_NAME`
> in Cell 1. The input video is auto-detected from
> `/content/drive/MyDrive/EcoCAR/video/input.mp4` (or any `.mp4` under an EcoCAR
> `video/` folder); change `VIDEO_IN` in Cell 5 if yours is elsewhere.

**Flow:** Cell 1 mount+deps+paths -> Cell 2 stage weights + val images -> Cell 3
load helpers (model/infer/draw) -> Cell 4 render >=8 image demos -> Cell 5
annotate a video.
"""

CELL1_MD = "### Cell 1: Mount Drive + deps + locate the NEWEST checkpoint"
CELL1 = '''\
import os, sys, subprocess
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
if not Path('/content/drive').exists():
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
if not os.path.isdir(REPO_ROOT):
    raise FileNotFoundError(f'Missing {REPO_ROOT} -- verify Drive sync.')
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for _pkg in ('addict', 'yapf'):
    try: __import__(_pkg)
    except ImportError: subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _pkg])
for _m in ('mmcv',):
    try: __import__(_m)
    except ImportError: subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _m])
try:
    import cv2  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python-headless'])

MIG       = Path(REPO_ROOT) / 'stage2/rmt_ppad_migration'
RMT       = MIG / 'vendor/RMT-PPAD'
DATASETS  = Path('/content/drive/MyDrive/EcoCAR/datasets')
FULL      = Path('/content/bdd_dataset')

# === The model under test ===========================================
# NB108 resumed the `laneiou_both_full` run; its best.pt is epoch 37 (the newest
# model). If your newest best.pt is under a different folder, change RUN_NAME.
CKPT_ROOT = Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints')
RUN_NAME  = 'laneiou_both_full'
CKPT_DIR  = CKPT_ROOT / RUN_NAME
# All demo outputs (images + annotated video) go HERE, next to the weights:
DEMO_DIR  = CKPT_DIR / 'demo'
DEMO_DIR.mkdir(parents=True, exist_ok=True)

print('[ok] repo =', MIG)
print('[ok] checkpoint dir:', CKPT_DIR, '| exists:', CKPT_DIR.exists())
for w in ('best.pt', 'last.pt'):
    print(f'   {w}:', (CKPT_DIR / w).exists())
print('[ok] demo outputs ->', DEMO_DIR)
if not CKPT_DIR.exists():
    print('[hint] folders present under', CKPT_ROOT, ':',
          sorted(p.name for p in CKPT_ROOT.glob("*"))[:30] if CKPT_ROOT.exists() else '<root missing>')
'''

# Cells 2 (data staging) and 3 (model/infer/draw/shim helpers) are reused
# VERBATIM from NB103 -- identical inference path to the curve-F1 metric.
CELL2_MD = nb103.CELL2_MD
CELL2 = nb103.CELL2
CELL3_MD = nb103.CELL3_MD
CELL3 = nb103.CELL3

CELL4_MD = """\
### Cell 4: Render >= 8 validation images (GT vs. best.pt) and save them

Two panels per image: ground truth (green lanes, red boxes) and the newest
best.pt prediction (yellow lanes, cyan boxes). Each prediction panel is also
saved as its own PNG in `DEMO_DIR` so you can drop the best ones straight into
the report's RMT-PPAD demonstration figure, alongside a combined grid.
"""
CELL4 = '''\
import matplotlib.pyplot as plt, random

N_SHOW = 8                                   # >= 8 demo images, per request
assert have_best, 'No best.pt found - cannot run the demo on the newest model.'
m_best = load_model(BEST)

val_dir = FULL/'images/val2017'
stems = sorted(p.stem for p in val_dir.glob('*.jpg'))
random.seed(7); pick = random.sample(stems, min(N_SHOW, len(stems)))
print('demo images:', pick)

fig, axes = plt.subplots(len(pick), 2, figsize=(12, 5*len(pick)))
if len(pick) == 1: axes = [axes]
saved = []
for r, st in enumerate(pick):
    img = cv2.imread(str(val_dir/f'{st}.jpg'))
    # GT panel
    g = draw(img, gt_lanes(st), [(*b,1,0) for b in gt_boxes(st)], lane_color=(0,255,0), box_color=(255,0,0))
    axes[r][0].imshow(g); axes[r][0].set_title(f'{st}  GT  (lanes={len(gt_lanes(st))}, boxes={len(gt_boxes(st))})', fontsize=9); axes[r][0].axis('off')
    # prediction panel (newest best.pt)
    L, B = infer(m_best, img)
    pred = draw(img, L, B)
    axes[r][1].imshow(pred); axes[r][1].set_title(f'best.pt (ep37)  lanes={len(L)} boxes={len(B)}', fontsize=9); axes[r][1].axis('off')
    # save the prediction panel on its own (BGR for cv2) for the report
    outp = DEMO_DIR / f'rmt_ppad_pred_{st}.png'
    cv2.imwrite(str(outp), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    saved.append(str(outp))
plt.tight_layout()
grid = DEMO_DIR / 'rmt_ppad_demo_grid.png'
plt.savefig(grid, dpi=90); plt.show()
print(f'[saved] grid -> {grid}')
print(f'[saved] {len(saved)} individual prediction PNGs -> {DEMO_DIR}')
for s in saved: print('   ', s)
print('READ: yellow pred-lanes should hug road markings; cyan boxes wrap vehicles. '
      'Pick the clearest few for the report figure.')
'''

CELL5_MD = """\
### Cell 5: Annotate a video with the newest model (saved to the checkpoint folder)

Runs best.pt frame-by-frame over `VIDEO_IN`, overlays predicted lanes (yellow) +
vehicle boxes (cyan), and writes an annotated MP4 to `DEMO_DIR` with a measured
FPS overlay. Mirrors the Stage-1 NB07 video approach (OpenCV read loop +
VideoWriter), adapted to the RMT-PPAD `infer`/`draw` helpers. The model runs at
640x640 (its train/eval resolution); each annotated frame is resized back to the
source resolution for the output video.
"""
CELL5 = '''\
import time, cv2, glob

# Input video on Drive. The clip lives at EcoCAR/video/input.mp4; try the known
# locations and, failing that, any .mp4 under an EcoCAR video/ folder.
_video_candidates = [
    '/content/drive/MyDrive/EcoCAR/video/input.mp4',
    '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/video/input.mp4',
]
VIDEO_IN = next((p for p in _video_candidates if Path(p).exists()), None)
if VIDEO_IN is None:
    hits = sorted(glob.glob('/content/drive/MyDrive/EcoCAR/video/*.mp4')) + \\
           sorted(glob.glob('/content/drive/MyDrive/EcoCAR/**/video/*.mp4', recursive=True))
    VIDEO_IN = hits[0] if hits else None
if not VIDEO_IN:
    raise FileNotFoundError(
        'No input video found. Tried:\\n  ' + '\\n  '.join(_video_candidates) +
        '\\nPlace your clip at /content/drive/MyDrive/EcoCAR/video/input.mp4 '
        '(or set VIDEO_IN to its path) and re-run.')
print(f'[video] using input: {VIDEO_IN}')

VIDEO_OUT = str(DEMO_DIR / 'rmt_ppad_demo.mp4')
MAX_FRAMES = 1200          # cap so a long clip doesn't run forever (~40s @30fps)
DRAW_FPS_TEXT = True

m = globals().get('m_best') or load_model(BEST)

cap = cv2.VideoCapture(VIDEO_IN)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
print(f'[video] in={VIDEO_IN}  {W}x{H} @ {src_fps:.1f}fps  frames~{n_total}')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, src_fps, (W, H))

n, infer_ms = 0, []
t_start = time.time()
while n < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()
    L, B = infer(m, frame)                       # decode lanes + boxes (640 frame)
    infer_ms.append((time.time() - t0) * 1000)
    ann_rgb = draw(frame, L, B)                  # RGB, 640x640
    ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
    ann_bgr = cv2.resize(ann_bgr, (W, H))        # back to source resolution
    if DRAW_FPS_TEXT and infer_ms:
        fps_now = 1000.0 / max(1e-6, sum(infer_ms[-30:]) / len(infer_ms[-30:]))
        cv2.putText(ann_bgr, f'RMT-PPAD ep37  {fps_now:5.1f} FPS (model)',
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    writer.write(ann_bgr)
    n += 1
    if n % 100 == 0:
        print(f'  ...{n} frames  (model {sum(infer_ms)/len(infer_ms):.1f} ms/frame)', flush=True)
cap.release(); writer.release()

wall = time.time() - t_start
mean_ms = sum(infer_ms) / max(1, len(infer_ms))
print(f'[done] wrote {n} annotated frames -> {VIDEO_OUT}')
print(f'[perf] model {mean_ms:.1f} ms/frame ({1000.0/max(1e-6,mean_ms):.1f} FPS, GPU, lanes+boxes decode incl.); '
      f'wall {wall:.1f}s end-to-end incl. video I/O')
print('NOTE: model FPS is the network forward + decode; end-to-end is lower due '
      'to CPU video read/encode (Stage-1 NB07 removed that bottleneck with a '
      'producer-consumer pipeline; here we keep it simple for a demo clip).')
'''


def build():
    nb = {
        "cells": [
            md(INTRO),
            md(CELL1_MD), code(CELL1),
            md(CELL2_MD), code(CELL2),
            md(CELL3_MD), code(CELL3),
            md(CELL4_MD), code(CELL4),
            md(CELL5_MD), code(CELL5),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / 'stage2_notebook_109_demo_images_and_video.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    build()
