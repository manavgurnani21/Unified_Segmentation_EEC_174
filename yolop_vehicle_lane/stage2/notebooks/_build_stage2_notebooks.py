"""Build Stage 2 Colab-safe notebooks 04-09.

Run from the repo root:
    python yolop_vehicle_lane/stage2/notebooks/_build_stage2_notebooks.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def _md(text: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'id': uuid.uuid4().hex[:12],
        'source': text.splitlines(keepends=True),
    }


def _code(text: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'id': uuid.uuid4().hex[:12],
        'source': text.splitlines(keepends=True),
    }


def _save(path: Path, cells: list, title: str) -> None:
    nb = {
        'cells': cells,
        'metadata': {
            'colab': {'provenance': [], 'name': title},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote: {path.name}')


# ---------------------------------------------------------------------------
# Shared helper cells
# ---------------------------------------------------------------------------
DRIVE_MOUNT = """from google.colab import drive
drive.mount('/content/drive')
!pip install -q yacs tqdm pyyaml opencv-python-headless tensorboard
"""

REPO_PATHS = """import os, sys, json, time, pathlib

REPO_ROOT     = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
DRIVE_ECOCAR  = '/content/drive/MyDrive/EcoCAR'
DRIVE_DATASETS = os.path.join(DRIVE_ECOCAR, 'datasets')
DRIVE_DOWNLOADS = os.path.join(DRIVE_ECOCAR, 'downloads')
DRIVE_TRAINING_RUNS = os.path.join(DRIVE_ECOCAR, 'training_runs')

os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print('REPO_ROOT =', REPO_ROOT)
"""


# ---------------------------------------------------------------------------
# 04 — Prepare CLRKD curve dataset (Colab-safe)
# ---------------------------------------------------------------------------
def build_04() -> None:
    cells = []
    cells.append(_md(
        '# Stage 2 / 04 — Prepare CLRKD curve dataset (Colab-safe)\n\n'
        'Goals:\n'
        '1. Build the CULane-style CLRKD curve label set from BDD100K JSON.\n'
        '2. Run an in-process diagnostic using `stage2.fusion.lane_targets` so we can\n'
        '   verify our DETR_GeoLane-derived parser before we trust the disk artifacts.\n'
        '3. Compress the result into a single tar archive and copy it back to Drive at:\n'
        '       /content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar\n\n'
        'This notebook DOES NOT train. It only prepares the curve dataset that the\n'
        'next notebooks consume.\n'
    ))
    cells.append(_code(DRIVE_MOUNT))
    cells.append(_code(REPO_PATHS))

    cells.append(_md('## 1. Locate / extract raw BDD100K\n'
                     '`stage2/scripts/00_prepare_rmt_dataset_links.py` already centralizes the\n'
                     'logic to extract `bdd100k_images_100k.zip` and the lane-tagged labels\n'
                     'archive from Drive. We simply re-use it.'))
    cells.append(_code(
        '# Pulls / extracts the raw BDD100K image + label trees to local /content.\n'
        '!python stage2/scripts/00_prepare_rmt_dataset_links.py \\\n'
        '    --dataset-root /content/bdd100k_vehicle5 \\\n'
        '    --raw-root /content/bdd100k_raw \\\n'
        '    --downloads-root {DRIVE_DOWNLOADS_PLACEHOLDER} \\\n'
        '    --output-root /content/bdd100k_clrkd_curve \\\n'
        '    --auto-extract\n'.replace('{DRIVE_DOWNLOADS_PLACEHOLDER}', '/content/drive/MyDrive/EcoCAR/downloads')
    ))

    cells.append(_md('## 2. CLRKDNet-style curve label generation\n'
                     'The existing `04_prepare_bdd_curve_labels.py` writes CULane-format\n'
                     '`.lines.txt` files plus train/val list files and per-image auxiliary\n'
                     'masks. Output goes under `/content/bdd100k_clrkd_curve/`.\n'))
    cells.append(_code(
        '!python stage2/scripts/04_prepare_bdd_curve_labels.py \\\n'
        '    --bdd-images /content/bdd100k_raw/100k \\\n'
        '    --bdd-labels /content/bdd100k_raw/100k \\\n'
        '    --output-root /content/bdd100k_clrkd_curve\n'
    ))

    cells.append(_md('## 3. In-process diagnostic with DETR_GeoLane-derived parser\n'
                     'This validates the BDD JSON parser using the new `stage2.fusion.lane_targets`\n'
                     'module. If counts here disagree with what the script produced, something is\n'
                     'wrong with one of the two paths.'))
    cells.append(_code(
        'from stage2.fusion.lane_targets import (\n'
        '    LaneLabelCache, LANE_TRAIN_CATS, LANE_CAT_TO_ID,\n'
        ')\n'
        '\n'
        'LABEL_DIR_TRAIN = \'/content/bdd100k_raw/100k/train\'\n'
        'LABEL_DIR_VAL   = \'/content/bdd100k_raw/100k/val\'\n'
        '\n'
        'cache_train = LaneLabelCache(LABEL_DIR_TRAIN, max_lanes=10, num_points=72)\n'
        'cache_val   = LaneLabelCache(LABEL_DIR_VAL,   max_lanes=10, num_points=72)\n'
        '\n'
        'diag_train = cache_train.diagnostics(sample_limit=3)\n'
        'diag_val   = cache_val.diagnostics(sample_limit=3)\n'
        '\n'
        'print(\'TRAIN diagnostics:\')\n'
        'for k, v in diag_train.items():\n'
        '    if k != \'examples\':\n'
        '        print(\'  \', k, \'=\', v)\n'
        'print(\'  example image names:\', [e[\\\'image\\\'] for e in diag_train[\\\'examples\\\']])\n'
        '\n'
        'print(\'\\nVAL diagnostics:\')\n'
        'for k, v in diag_val.items():\n'
        '    if k != \'examples\':\n'
        '        print(\'  \', k, \'=\', v)\n'
    ))

    cells.append(_md('## 4. Visualize a few samples\n'
                     'Plot the original image, dense lane points, and a rendered soft mask for\n'
                     'sanity-checking that the curve parsing is producing usable supervision.'))
    cells.append(_code(
        'import matplotlib.pyplot as plt\n'
        'import numpy as np\n'
        'import cv2\n'
        '\n'
        'from stage2.fusion.lane_targets import soft_polyline_mask_numpy\n'
        '\n'
        'IMG_DIR_TRAIN = \'/content/bdd100k_raw/100k/train\'\n'
        '\n'
        'sample_names = [e[\\\'image\\\'] for e in cache_train.diagnostics(sample_limit=4)[\\\'examples\\\']]\n'
        'fig, axes = plt.subplots(len(sample_names), 3, figsize=(14, 4*len(sample_names)))\n'
        'if len(sample_names) == 1:\n'
        '    axes = axes[None, :]\n'
        '\n'
        'for row, name in enumerate(sample_names):\n'
        '    targets = cache_train.get(name)\n'
        '    if targets is None:\n'
        '        continue\n'
        '    img_path = os.path.join(IMG_DIR_TRAIN, name)\n'
        '    if not os.path.exists(img_path):\n'
        '        continue\n'
        '    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)\n'
        '    H, W = img.shape[:2]\n'
        '    overlay = img.copy()\n'
        '    for li in range(targets[\\\'existence\\\'].shape[0]):\n'
        '        if targets[\\\'existence\\\'][li] < 1: continue\n'
        '        pts = targets[\\\'points\\\'][li] * np.array([W, H], dtype=np.float32)\n'
        '        vis = targets[\\\'visibility\\\'][li]\n'
        '        valid = pts[vis > 0.5].astype(np.int32)\n'
        '        for (x, y) in valid:\n'
        '            cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)\n'
        '    soft = soft_polyline_mask_numpy(targets[\\\'points\\\'], targets[\\\'visibility\\\'], height=72, width=128)\n'
        '    axes[row, 0].imshow(img); axes[row, 0].set_title(name); axes[row, 0].axis(\\\'off\\\')\n'
        '    axes[row, 1].imshow(overlay); axes[row, 1].set_title(\\\'extracted points\\\'); axes[row, 1].axis(\\\'off\\\')\n'
        '    axes[row, 2].imshow(soft, cmap=\\\'magma\\\'); axes[row, 2].set_title(\\\'aux mask\\\'); axes[row, 2].axis(\\\'off\\\')\n'
        'plt.tight_layout(); plt.show()\n'
    ))

    cells.append(_md('## 5. Pack the prepared dataset into a tar archive on Drive'))
    cells.append(_code(
        'import shutil, subprocess\n'
        '\n'
        'OUT_TAR = os.path.join(DRIVE_DATASETS, \\\'bdd100k_clrkd_curve.tar\\\')\n'
        'os.makedirs(DRIVE_DATASETS, exist_ok=True)\n'
        '\n'
        '# Pack only the files the trainer needs: list/, lines/, masks/, optionally images/.\n'
        'PACK_ROOT = \\\'/content/bdd100k_clrkd_curve\\\'\n'
        'assert os.path.isdir(PACK_ROOT), f\\\'expected output dir at {PACK_ROOT}\\\'\n'
        '\n'
        'STAGING = \\\'/content/_pack_clrkd_curve\\\'\n'
        'if os.path.exists(STAGING):\n'
        '    shutil.rmtree(STAGING)\n'
        'os.makedirs(STAGING, exist_ok=True)\n'
        '\n'
        'for sub in [\\\'list\\\', \\\'lines\\\', \\\'masks\\\']:\n'
        '    src = os.path.join(PACK_ROOT, sub)\n'
        '    if os.path.exists(src):\n'
        '        os.symlink(src, os.path.join(STAGING, sub))\n'
        '\n'
        'cmd = [\\\'tar\\\', \\\'-cf\\\', OUT_TAR, \\\'-C\\\', STAGING, \\\'.\\\']\n'
        'print(\\\' \\\'.join(cmd))\n'
        'subprocess.check_call(cmd)\n'
        'print(\\\'Wrote\\\', OUT_TAR, os.path.getsize(OUT_TAR) // (1024*1024), \\\'MB\\\')\n'
        'print(\\\'Drive output path:\\\', OUT_TAR)\n'
    ))

    _save(NB_DIR / '04_prepare_clrkd_curve_dataset_colab.ipynb', cells, 'Stage 2 / 04 prepare CLRKD curve')


# ---------------------------------------------------------------------------
# 05 — Train CLRKD lane-only baseline (vendored CLRKDNet path)
# ---------------------------------------------------------------------------
def build_05() -> None:
    cells = []
    cells.append(_md(
        '# Stage 2 / 05 — Train CLRKD curve lane-only baseline\n\n'
        'Trains the **upstream CLRKDNet** lane head on the BDD100K curve dataset\n'
        'prepared by notebook 04. This is the strict reproduction path; the\n'
        'in-house lighter head is exercised in notebook 06.\n\n'
        'Output saved to Drive as a single tar at:\n'
        '`/content/drive/MyDrive/EcoCAR/training_runs/stage2_clrkd_lane_only.tar`.\n'
    ))
    cells.append(_code(DRIVE_MOUNT))
    cells.append(_code(REPO_PATHS))

    cells.append(_md('## 1. Extract the prepared curve dataset from Drive'))
    cells.append(_code(
        'import shutil, subprocess\n'
        '\n'
        'CLRKD_TAR = os.path.join(DRIVE_DATASETS, \\\'bdd100k_clrkd_curve.tar\\\')\n'
        'CLRKD_LOCAL = \\\'/content/bdd100k_clrkd_curve\\\'\n'
        '\n'
        'if not os.path.exists(CLRKD_TAR):\n'
        '    raise FileNotFoundError(f\\\'Missing {CLRKD_TAR}. Run notebook 04 first.\\\')\n'
        '\n'
        'os.makedirs(CLRKD_LOCAL, exist_ok=True)\n'
        'subprocess.check_call([\\\'tar\\\', \\\'-xf\\\', CLRKD_TAR, \\\'-C\\\', CLRKD_LOCAL])\n'
        '\n'
        '# Sanity\n'
        'for sub in [\\\'list\\\', \\\'lines\\\']:\n'
        '    p = os.path.join(CLRKD_LOCAL, sub)\n'
        '    print(sub, \\\'->\\\', os.path.exists(p), \\\'count\\\', sum(1 for _ in pathlib.Path(p).rglob(\\\'*\\\')) if os.path.exists(p) else 0)\n'
    ))

    cells.append(_md('## 2. Loud preflight'))
    cells.append(_code(
        'CONFIG = os.path.join(REPO_ROOT, \\\'stage2\\\', \\\'configs\\\', \\\'BDD100K_CLRKD_Curve.py\\\')\n'
        'WORK_DIR = \\\'/content/clrkd_lane_only_run\\\'\n'
        'OUT_TAR  = os.path.join(DRIVE_TRAINING_RUNS, \\\'stage2_clrkd_lane_only.tar\\\')\n'
        '\n'
        'os.makedirs(WORK_DIR, exist_ok=True)\n'
        'os.makedirs(DRIVE_TRAINING_RUNS, exist_ok=True)\n'
        '\n'
        'assert os.path.exists(CONFIG), f\\\'Missing config: {CONFIG}\\\'\n'
        'assert os.path.exists(\\\'/content/bdd100k_clrkd_curve/list\\\'), \\\'Curve list missing — re-run nb 04.\\\'\n'
        '\n'
        'print(\\\'=\\\' * 60)\n'
        'print(\\\'[Stage2 / 05 preflight]\\\')\n'
        'print(\\\'  Config         :\\\', CONFIG)\n'
        'print(\\\'  Curve dataset  : /content/bdd100k_clrkd_curve\\\')\n'
        'print(\\\'  Work dir       :\\\', WORK_DIR)\n'
        'print(\\\'  Output tar     :\\\', OUT_TAR)\n'
        'print(\\\'=\\\' * 60)\n'
    ))

    cells.append(_md('## 3. Train via the existing wrapper'))
    cells.append(_code(
        '!python stage2/scripts/05_train_bdd_clrkd_curve.py \\\n'
        '    --project-root /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane \\\n'
        '    --config /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/configs/BDD100K_CLRKD_Curve.py \\\n'
        '    --gpus 0 \\\n'
        '    --work-dirs /content/clrkd_lane_only_run\n'
    ))

    cells.append(_md('## 4. Pack outputs and push to Drive'))
    cells.append(_code(
        'import subprocess\n'
        'cmd = [\\\'tar\\\', \\\'-cf\\\', OUT_TAR, \\\'-C\\\', WORK_DIR, \\\'.\\\']\n'
        'subprocess.check_call(cmd)\n'
        'print(\\\'Wrote\\\', OUT_TAR, os.path.getsize(OUT_TAR) // (1024*1024), \\\'MB\\\')\n'
    ))

    _save(NB_DIR / '05_train_clrkd_curve_baseline_colab.ipynb', cells, 'Stage 2 / 05 train CLRKD lane-only')


# ---------------------------------------------------------------------------
# 06 — Basic RMT-detection + CLRKD-lane fusion train (in-house head)
# ---------------------------------------------------------------------------
def build_06() -> None:
    cells = []
    cells.append(_md(
        '# Stage 2 / 06 — Basic RMT-style detection + CLRKD curve lane fusion\n\n'
        '**Conservative first fusion.** Detection branch comes from the vendored RMT-PPAD\n'
        'RT-DETR decoder; lane branch is the in-house `CurveLaneHead`. Backbone stays at\n'
        'the existing CSP backbone for this run so the only new piece is the curve lane head.\n'
        'Loss: `L_total = L_det + lambda_lane * L_lane`.\n\n'
        'Output: `/content/drive/MyDrive/EcoCAR/training_runs/stage2_rmt_clrkd_basic_fusion.tar`.\n'
    ))
    cells.append(_code(DRIVE_MOUNT))
    cells.append(_code(REPO_PATHS))

    cells.append(_md('## 1. Extract curve dataset + raw BDD images'))
    cells.append(_code(
        'import subprocess\n'
        'CLRKD_TAR = os.path.join(DRIVE_DATASETS, \\\'bdd100k_clrkd_curve.tar\\\')\n'
        'CLRKD_LOCAL = \\\'/content/bdd100k_clrkd_curve\\\'\n'
        'os.makedirs(CLRKD_LOCAL, exist_ok=True)\n'
        'if not os.path.exists(os.path.join(CLRKD_LOCAL, \\\'list\\\')):\n'
        '    subprocess.check_call([\\\'tar\\\', \\\'-xf\\\', CLRKD_TAR, \\\'-C\\\', CLRKD_LOCAL])\n'
        '\n'
        '# Raw BDD images (use stage1 dataset prep helper).\n'
        'from lib.utils.drive_dataset import (\n'
        '    ensure_local_dataset_from_drive, find_raw_bdd_root,\n'
        '    resolve_bdd_images_100k_dir, resolve_bdd_labels_100k_dir,\n'
        ')\n'
        'DATASET_ROOT = ensure_local_dataset_from_drive(\\\'bdd100k_vehicle5\\\', DRIVE_ECOCAR)\n'
        'RAW_BDD_ROOT = find_raw_bdd_root(DRIVE_ECOCAR)\n'
        'BDD_IMAGES   = resolve_bdd_images_100k_dir(RAW_BDD_ROOT, DRIVE_ECOCAR)\n'
        'print(\\\'BDD images    :\\\', BDD_IMAGES)\n'
        'print(\\\'Curve dataset :\\\', CLRKD_LOCAL)\n'
    ))

    cells.append(_md('## 2. Load fusion config + build model'))
    cells.append(_code(
        'import yaml, torch, torch.nn as nn\n'
        '\n'
        'CFG_PATH = os.path.join(REPO_ROOT, \\\'stage2\\\', \\\'configs\\\', \\\'rmt_clrkd_basic_fusion.yaml\\\')\n'
        'with open(CFG_PATH) as fh:\n'
        '    fcfg = yaml.safe_load(fh)\n'
        '\n'
        '# CSP backbone is the same one Stage 1 uses; we only swap the lane head.\n'
        'from lib.models.common import C3, Conv\n'
        'from stage2.fusion.lane_head import CurveLaneHead\n'
        'from stage2.fusion.losses import FusionLaneLoss, FusionLossConfig, UncertaintyMultiTaskLoss\n'
        '\n'
        'class TinyCSPBackbone(nn.Module):\n'
        '    """A trimmed feature backbone for fusion experiments. Outputs P3, P4, P5."""\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        '        self.stem  = Conv(3, 32, 6, 2, 2)\n'
        '        self.down1 = Conv(32, 64, 3, 2)\n'
        '        self.c3_1  = C3(64, 64, 1)\n'
        '        self.down2 = Conv(64, 128, 3, 2)\n'
        '        self.c3_2  = C3(128, 128, 2)\n'
        '        self.down3 = Conv(128, 256, 3, 2)\n'
        '        self.c3_3  = C3(256, 256, 2)\n'
        '        self.down4 = Conv(256, 512, 3, 2)\n'
        '        self.c3_4  = C3(512, 512, 1)\n'
        '    def forward(self, x):\n'
        '        x = self.stem(x); x = self.down1(x); x = self.c3_1(x)\n'
        '        x = self.down2(x); p3 = self.c3_2(x)\n'
        '        x = self.down3(p3); p4 = self.c3_3(x)\n'
        '        x = self.down4(p4); p5 = self.c3_4(x)\n'
        '        return [p3, p4, p5]\n'
        '\n'
        'backbone = TinyCSPBackbone()\n'
        'lane_head = CurveLaneHead(\n'
        '    in_channels=fcfg[\\\'model\\\'][\\\'feature_channels\\\'],\n'
        '    embed_dim=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'embed_dim\\\'],\n'
        '    max_lanes=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'max_lanes\\\'],\n'
        '    num_points=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'num_points\\\'],\n'
        '    mask_size=tuple(fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'mask_size\\\']),\n'
        '    mask_aux=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'mask_aux\\\'],\n'
        '    num_lane_classes=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'num_lane_classes\\\'],\n'
        ')\n'
        '\n'
        '# Detection head: for the basic-fusion baseline we keep the existing YOLOPX detection\n'
        '# head wrapper around the same feature stack. Importing the decoder here would pull in\n'
        '# the entire vendored RMT-PPAD trainer, which is heavier than this notebook needs. The\n'
        '# RMT decoder is exercised in notebook 07.\n'
        'detection_head = None  # placeholder; wire to YOLOPX or RT-DETR decoder per experiment\n'
        '\n'
        'lane_loss_cfg = FusionLossConfig(\n'
        '    lambda_lane=fcfg[\\\'loss\\\'][\\\'lambda_lane\\\'],\n'
        '    use_uncertainty=fcfg[\\\'loss\\\'][\\\'use_uncertainty\\\'],\n'
        '    w_cls=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'w_cls\\\'],\n'
        '    w_reg=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'w_reg\\\'],\n'
        '    w_iou=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'w_iou\\\'],\n'
        '    w_mask=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'w_mask\\\'],\n'
        '    w_smooth=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'w_smooth\\\'],\n'
        '    focal_alpha=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'focal_alpha\\\'],\n'
        '    focal_gamma=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'focal_gamma\\\'],\n'
        '    line_iou_radius=fcfg[\\\'loss\\\'][\\\'lane\\\'][\\\'line_iou_radius\\\'],\n'
        ')\n'
        'lane_loss = FusionLaneLoss(lane_loss_cfg)\n'
        '\n'
        'device = torch.device(\\\'cuda\\\' if torch.cuda.is_available() else \\\'cpu\\\')\n'
        'backbone = backbone.to(device); lane_head = lane_head.to(device)\n'
        'param_count = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in lane_head.parameters())\n'
        'print(\\\'=\\\' * 60)\n'
        'print(\\\'[Stage2 / 06 preflight]\\\')\n'
        'print(\\\'  Config       :\\\', CFG_PATH)\n'
        'print(\\\'  Backbone     : TinyCSPBackbone (P3,P4,P5)\\\')\n'
        'print(\\\'  Lane head    : CurveLaneHead\\\')\n'
        'print(\\\'  Detection    : (deferred to nb 07)\\\')\n'
        'print(\\\'  Total params :\\\', param_count // 1000, \\\'k\\\')\n'
        'print(\\\'=\\\' * 60)\n'
    ))

    cells.append(_md('## 3. Build dataloader (lane-only training first)\n'
                     'For this conservative baseline we train just the lane branch on top of the CSP\n'
                     'backbone; the detection branch is wired in notebook 07. This makes the lane\n'
                     'metric directly comparable to nb 05 while exercising the in-house lane head.'))
    cells.append(_code(
        '# Lightweight dataset adapter: image -> (image_tensor, targets_dict).\n'
        'import cv2, numpy as np, torch\n'
        'from torch.utils.data import Dataset, DataLoader\n'
        'from stage2.fusion.lane_targets import LaneLabelCache, soft_polyline_mask_numpy\n'
        '\n'
        'class StageTwoCurveDataset(Dataset):\n'
        '    def __init__(self, image_dir, label_dir, max_lanes=10, num_points=72,\n'
        '                 image_hw=(384, 640), aux_mask_size=(72, 128)):\n'
        '        self.image_dir = image_dir\n'
        '        self.cache = LaneLabelCache(label_dir, max_lanes=max_lanes, num_points=num_points)\n'
        '        self.image_hw = image_hw\n'
        '        self.aux_mask_size = aux_mask_size\n'
        '        self.names = [n for n in self.cache.keys()\n'
        '                      if os.path.exists(os.path.join(image_dir, n)) and self.cache.has_lanes(n)]\n'
        '    def __len__(self): return len(self.names)\n'
        '    def __getitem__(self, idx):\n'
        '        name = self.names[idx]\n'
        '        img = cv2.imread(os.path.join(self.image_dir, name))\n'
        '        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n'
        '        img = cv2.resize(img, (self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_LINEAR)\n'
        '        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)\n'
        '        targets = self.cache.get(name)\n'
        '        mask = soft_polyline_mask_numpy(\n'
        '            targets[\\\'points\\\'], targets[\\\'visibility\\\'],\n'
        '            height=self.aux_mask_size[0], width=self.aux_mask_size[1],\n'
        '        )\n'
        '        return img_t, {\n'
        '            \\\'existence\\\': torch.from_numpy(targets[\\\'existence\\\']),\n'
        '            \\\'points\\\': torch.from_numpy(targets[\\\'points\\\']),\n'
        '            \\\'visibility\\\': torch.from_numpy(targets[\\\'visibility\\\']),\n'
        '            \\\'mask_target\\\': torch.from_numpy(mask).unsqueeze(0),\n'
        '            \\\'lane_type\\\': torch.from_numpy(targets[\\\'lane_type\\\']),\n'
        '        }\n'
        '\n'
        'def lane_collate(batch):\n'
        '    imgs = torch.stack([b[0] for b in batch], 0)\n'
        '    keys = batch[0][1].keys()\n'
        '    out  = {k: torch.stack([b[1][k] for b in batch], 0) for k in keys}\n'
        '    return imgs, out\n'
        '\n'
        'IMG_TRAIN  = os.path.join(BDD_IMAGES, \\\'train\\\')\n'
        'IMG_VAL    = os.path.join(BDD_IMAGES, \\\'val\\\')\n'
        'LBL_TRAIN  = \\\'/content/bdd100k_raw/100k/train\\\'\n'
        'LBL_VAL    = \\\'/content/bdd100k_raw/100k/val\\\'\n'
        '\n'
        'train_ds = StageTwoCurveDataset(IMG_TRAIN, LBL_TRAIN)\n'
        'val_ds   = StageTwoCurveDataset(IMG_VAL,   LBL_VAL)\n'
        'print(\\\'train\\\', len(train_ds), \\\'val\\\', len(val_ds))\n'
        '\n'
        'BS = fcfg[\\\'train\\\'][\\\'batch_size\\\']\n'
        'train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=fcfg[\\\'train\\\'][\\\'workers\\\'],\n'
        '                          pin_memory=True, collate_fn=lane_collate)\n'
        'val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=fcfg[\\\'train\\\'][\\\'workers\\\'],\n'
        '                          pin_memory=True, collate_fn=lane_collate)\n'
    ))

    cells.append(_md('## 4. Train loop (lane-only, conservative baseline)'))
    cells.append(_code(
        'import torch.optim as optim\n'
        'from torch.cuda.amp import autocast, GradScaler\n'
        '\n'
        'opt = optim.AdamW(list(backbone.parameters()) + list(lane_head.parameters()),\n'
        '                  lr=fcfg[\\\'train\\\'][\\\'lr0\\\'],\n'
        '                  weight_decay=fcfg[\\\'train\\\'][\\\'weight_decay\\\'])\n'
        'scaler = GradScaler()\n'
        '\n'
        'CKPT_DIR = \\\'/content/stage2_basic_fusion_run\\\'\n'
        'os.makedirs(CKPT_DIR, exist_ok=True)\n'
        'OUT_TAR = os.path.join(DRIVE_TRAINING_RUNS, \\\'stage2_rmt_clrkd_basic_fusion.tar\\\')\n'
        'os.makedirs(DRIVE_TRAINING_RUNS, exist_ok=True)\n'
        '\n'
        'best_val = float(\\\'inf\\\')\n'
        'for epoch in range(fcfg[\\\'train\\\'][\\\'end_epoch\\\']):\n'
        '    backbone.train(); lane_head.train()\n'
        '    epoch_loss = 0.0; n = 0\n'
        '    for img, tgt in train_loader:\n'
        '        img = img.to(device, non_blocking=True)\n'
        '        tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}\n'
        '        opt.zero_grad(set_to_none=True)\n'
        '        with autocast():\n'
        '            feats = backbone(img)\n'
        '            pred  = lane_head(feats)\n'
        '            loss, comp = lane_loss(pred, tgt)\n'
        '        scaler.scale(loss).backward()\n'
        '        scaler.unscale_(opt)\n'
        '        torch.nn.utils.clip_grad_norm_(\n'
        '            list(backbone.parameters()) + list(lane_head.parameters()),\n'
        '            fcfg[\\\'train\\\'][\\\'grad_clip_norm\\\'],\n'
        '        )\n'
        '        scaler.step(opt); scaler.update()\n'
        '        epoch_loss += float(loss.detach()); n += 1\n'
        '    avg = epoch_loss / max(n, 1)\n'
        '\n'
        '    backbone.eval(); lane_head.eval()\n'
        '    val_loss = 0.0; vn = 0\n'
        '    with torch.no_grad():\n'
        '        for img, tgt in val_loader:\n'
        '            img = img.to(device, non_blocking=True)\n'
        '            tgt = {k: v.to(device, non_blocking=True) for k, v in tgt.items()}\n'
        '            feats = backbone(img)\n'
        '            pred  = lane_head(feats)\n'
        '            l, _  = lane_loss(pred, tgt)\n'
        '            val_loss += float(l); vn += 1\n'
        '    val_avg = val_loss / max(vn, 1)\n'
        '\n'
        '    print(f\\\'epoch {epoch:03d}  train={avg:.4f}  val={val_avg:.4f}\\\')\n'
        '\n'
        '    state = {\\\'backbone\\\': backbone.state_dict(), \\\'lane_head\\\': lane_head.state_dict(),\n'
        '             \\\'optimizer\\\': opt.state_dict(), \\\'epoch\\\': epoch,\n'
        '             \\\'val_loss\\\': val_avg, \\\'config\\\': fcfg}\n'
        '    torch.save(state, os.path.join(CKPT_DIR, \\\'latest.pth\\\'))\n'
        '    if val_avg < best_val:\n'
        '        best_val = val_avg\n'
        '        torch.save(state, os.path.join(CKPT_DIR, \\\'best.pth\\\'))\n'
        '        print(\\\'  ** new best val loss\\\', best_val)\n'
        '\n'
        'import subprocess\n'
        'subprocess.check_call([\\\'tar\\\', \\\'-cf\\\', OUT_TAR, \\\'-C\\\', CKPT_DIR, \\\'.\\\'])\n'
        'print(\\\'wrote\\\', OUT_TAR, os.path.getsize(OUT_TAR) // (1024 * 1024), \\\'MB\\\')\n'
    ))

    _save(NB_DIR / '06_train_rmt_clrkd_basic_fusion_colab.ipynb', cells, 'Stage 2 / 06 basic RMT+CLRKD fusion')


# ---------------------------------------------------------------------------
# 07 — RMT backbone + CLRKD curve head (skeleton)
# ---------------------------------------------------------------------------
def build_07() -> None:
    cells = []
    cells.append(_md(
        '# Stage 2 / 07 — RMT backbone + RT-DETR detection head + CLRKD curve lane head\n\n'
        'This notebook moves to the heavier fusion: vendored RMT-PPAD backbone & encoder\n'
        '(`stage2/vendor/RMT-PPAD/ultralytics/...`) + RT-DETR-style detection decoder, with\n'
        'the in-house `CurveLaneHead` consuming the lower / mid-level features (lanes are\n'
        'thin and local, so we route P2/P3 to lane and P4/P5 to detection).\n\n'
        '**Status**: skeleton — the RMT vendor pulls in a sizeable Ultralytics tree, so\n'
        'the actual `nn.Module` instantiation requires `PYTHONPATH` to include the vendor\n'
        'root. Cells below set this up; the actual training body mirrors notebook 06 but\n'
        'with the detection head wired through.\n'
    ))
    cells.append(_code(DRIVE_MOUNT))
    cells.append(_code(REPO_PATHS))

    cells.append(_md('## 1. Add vendored RMT-PPAD to PYTHONPATH'))
    cells.append(_code(
        'VENDOR_RMT = os.path.join(REPO_ROOT, \\\'stage2\\\', \\\'vendor\\\', \\\'RMT-PPAD\\\')\n'
        'assert os.path.exists(VENDOR_RMT), f\\\'Missing vendored RMT-PPAD at {VENDOR_RMT}\\\'\n'
        'if VENDOR_RMT not in sys.path:\n'
        '    sys.path.insert(0, VENDOR_RMT)\n'
        '\n'
        '# Smoke import to surface any missing dependency.\n'
        'from ultralytics.nn.modules import Conv, ResNetLayer, AIFI\n'
        'from ultralytics.nn.modules.head import MTDETRDecoder\n'
        'print(\\\'RMT vendor importable: OK\\\')\n'
    ))

    cells.append(_md('## 2. Build the joint model\n'
                     '`stage2/fusion/model.py:FusionModel` accepts any backbone returning a list of\n'
                     'feature maps and any detection head module. Below we build a thin RMT-style\n'
                     'backbone and feed P2, P3 to the lane head while routing P3, P4, P5 to the\n'
                     'detection head.\n\n'
                     '*TODO when running for the first time*: confirm the channel sizes the actual\n'
                     '`ResNetLayer` outputs match the lane head\\\'s `in_channels` argument; the dummy\n'
                     'forward in the next cell will fail loudly otherwise.'))
    cells.append(_code(
        '# Skeleton: build_rmt_backbone / build_rt_detr_head are project-level glue we\n'
        '# implement here so the rest of the notebook stays declarative. They are tiny\n'
        '# wrappers around the vendored RMT modules and can be lifted into\n'
        '# stage2/fusion/rmt_adapter.py after this notebook is verified end-to-end.\n'
        'import torch.nn as nn\n'
        'from stage2.fusion.lane_head import CurveLaneHead\n'
        'from stage2.fusion.model import FusionModel\n'
        '\n'
        'class RMTBackboneWrapper(nn.Module):\n'
        '    """Slim wrapper around vendored RMT ResNetLayer that returns a list of scales."""\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        '        from ultralytics.nn.modules import Conv, ResNetLayer\n'
        '        self.stem = Conv(3, 32, 6, 2, 2)\n'
        '        self.s1   = ResNetLayer(32,  64, 1, 1, 1)\n'
        '        self.s2   = ResNetLayer(64, 128, 2, 1, 2)\n'
        '        self.s3   = ResNetLayer(128, 256, 2, 1, 2)\n'
        '        self.s4   = ResNetLayer(256, 512, 2, 1, 2)\n'
        '    def forward(self, x):\n'
        '        x = self.stem(x); x = self.s1(x)\n'
        '        p2 = self.s2(x); p3 = self.s3(p2); p4 = self.s4(p3)\n'
        '        return [p2, p3, p4]\n'
        '\n'
        'backbone = RMTBackboneWrapper()\n'
        '# Lane head consumes the higher-resolution P2/P3 (low semantic, high detail).\n'
        'lane_head = CurveLaneHead(in_channels=[128, 256], embed_dim=128, max_lanes=10, num_points=72)\n'
        'fusion = FusionModel(backbone=backbone, feature_channels=[128, 256, 512],\n'
        '                     lane_head=lane_head, lane_in_indices=[0, 1])\n'
        '\n'
        'import torch\n'
        'device = torch.device(\\\'cuda\\\' if torch.cuda.is_available() else \\\'cpu\\\')\n'
        'fusion = fusion.to(device)\n'
        'with torch.no_grad():\n'
        '    out = fusion(torch.randn(1, 3, 384, 640, device=device))\n'
        '    print(\\\'lane.cls_logits :\\\', tuple(out[\\\'lane\\\'][\\\'cls_logits\\\'].shape))\n'
        '    print(\\\'lane.coord_pred :\\\', tuple(out[\\\'lane\\\'][\\\'coord_pred\\\'].shape))\n'
    ))

    cells.append(_md('## 3. Training body\n'
                     'Reuse the dataloader and loss from notebook 06; difference is the model is\n'
                     'now `FusionModel(...)` instead of `(backbone, lane_head)` separately. The\n'
                     'detection branch is wired by passing `detection_head=` into `FusionModel`.\n'
                     'Detection-side loss: import RMT\\\'s `MTDETRDLoss` from\n'
                     '`ultralytics.models.utils.loss` and combine via `L_total = L_det + lambda_lane * L_lane`.\n\n'
                     '*This cell is left as a TODO so you can tune RT-DETR-side hyperparameters\n'
                     'before committing to a long training run.*'))
    cells.append(_code(
        '# TODO: instantiate RT-DETR detection head + matching loss, then run the\n'
        '# train loop body from notebook 06 with the combined loss.\n'
        '#\n'
        '# from ultralytics.nn.modules.head import MTDETRDecoder\n'
        '# from ultralytics.models.utils.loss import MTDETRDLoss\n'
        '# det_head = MTDETRDecoder(...)\n'
        '# det_loss = MTDETRDLoss(...)\n'
        '#\n'
        '# Then: total = det_loss(...) + fcfg[\\\'loss\\\'][\\\'lambda_lane\\\'] * lane_loss(...)\n'
        'print(\\\'Skeleton — fill in detection wiring before launching long training.\\\')\n'
    ))

    _save(NB_DIR / '07_train_rmt_backbone_clrkd_curve_head_colab.ipynb', cells, 'Stage 2 / 07 RMT backbone + CLRKD head')


# ---------------------------------------------------------------------------
# 08 — YOLO26 backbone/neck experiment (skeleton)
# ---------------------------------------------------------------------------
def build_08() -> None:
    cells = []
    cells.append(_md(
        '# Stage 2 / 08 — YOLO26 backbone/neck + CLRKD curve lane head (experimental)\n\n'
        'Uses the sister project `yolo26_pipeline` for backbone/neck construction. We slice\n'
        'the YOLO26 model up to (but not including) the detection head, then attach the\n'
        'in-house `CurveLaneHead` to its FPN outputs. **Experimental** — kept off the stable\n'
        'Stage 2 baseline.\n'
    ))
    cells.append(_code(DRIVE_MOUNT))
    cells.append(_code(REPO_PATHS))

    cells.append(_md('## 1. Locate the YOLO26 pipeline'))
    cells.append(_code(
        'YOLO26_DIR = os.path.join(\\\'/content/drive/MyDrive/EcoCAR\\\', \\\'..\\\', \\\'yolo26_pipeline\\\')\n'
        'YOLO26_DIR = os.path.normpath(YOLO26_DIR)\n'
        '# In Colab, the workspace is typically not the same as the local clone path. Adjust as needed.\n'
        '# If you cloned EcoCAR-Perception-Pipeline-YOLO26-BDD100K to Drive, point YOLO26_DIR to:\n'
        '#   /content/drive/MyDrive/EcoCAR-Perception-Pipeline-YOLO26-BDD100K/yolo26_pipeline\n'
        'print(\\\'YOLO26_DIR =\\\', YOLO26_DIR, \\\'exists=\\\', os.path.exists(YOLO26_DIR))\n'
        '\n'
        'YOLO26_SRC = os.path.join(YOLO26_DIR, \\\'src\\\')\n'
        'if YOLO26_SRC not in sys.path and os.path.exists(YOLO26_SRC):\n'
        '    sys.path.insert(0, YOLO26_SRC)\n'
    ))

    cells.append(_md('## 2. Build the joint model'))
    cells.append(_code(
        '# from multitask_model import build_multitask_model\n'
        '# m = build_multitask_model(cfg={\\\'model\\\': {\\\'backbone_weights\\\': \\\'yolo26s.pt\\\'}})\n'
        '#\n'
        '# Pull the backbone+neck out of m and wrap it in stage2.fusion.model.FusionModel.\n'
        '# Channel sizes are auto-probed by `_find_neck_output_indices` inside MultiTaskYOLO\n'
        '# but we need to expose them — the cleanest approach is to call m._probe_channels()\n'
        '# (or whatever the project provides) and feed those to CurveLaneHead.\n'
        'print(\\\'Skeleton — see stage2/STAGE2_PLAN.md for the integration plan.\\\')\n'
    ))

    _save(NB_DIR / '08_yolo26_backbone_neck_experiment_colab.ipynb', cells, 'Stage 2 / 08 YOLO26 backbone experiment')


# ---------------------------------------------------------------------------
# 09 — Stage 2 evaluation + video profile
# ---------------------------------------------------------------------------
def build_09() -> None:
    cells = []
    cells.append(_md(
        '# Stage 2 / 09 — Stage 2 evaluation, qualitative vis & video profile\n\n'
        'Evaluates a Stage 2 fusion checkpoint and produces:\n'
        '- detection mAP50 / mAP50-95 / recall (if det head present)\n'
        '- lane existence accuracy, soft-mask IoU, LineIoU\n'
        '- per-stage timing (preprocess / backbone / lane / det / postprocess)\n'
        '- qualitative visualizations\n'
        '- optional video overlay\n\n'
        'Defaults to the basic-fusion checkpoint produced by notebook 06.\n'
    ))
    cells.append(_code(DRIVE_MOUNT))
    cells.append(_code(REPO_PATHS))

    cells.append(_md('## 1. Loud preflight'))
    cells.append(_code(
        'CHECKPOINT_TAR = os.path.join(DRIVE_TRAINING_RUNS, \\\'stage2_rmt_clrkd_basic_fusion.tar\\\')\n'
        'CHECKPOINT_LOCAL = \\\'/content/stage2_basic_fusion_eval\\\'\n'
        'os.makedirs(CHECKPOINT_LOCAL, exist_ok=True)\n'
        'assert os.path.exists(CHECKPOINT_TAR), f\\\'Missing {CHECKPOINT_TAR}. Run notebook 06 first.\\\'\n'
        '\n'
        'import subprocess\n'
        'subprocess.check_call([\\\'tar\\\', \\\'-xf\\\', CHECKPOINT_TAR, \\\'-C\\\', CHECKPOINT_LOCAL])\n'
        '\n'
        'CKPT = os.path.join(CHECKPOINT_LOCAL, \\\'best.pth\\\')\n'
        'if not os.path.exists(CKPT):\n'
        '    CKPT = os.path.join(CHECKPOINT_LOCAL, \\\'latest.pth\\\')\n'
        'assert os.path.exists(CKPT), f\\\'No checkpoint inside {CHECKPOINT_TAR}\\\'\n'
        '\n'
        'print(\\\'=\\\' * 60)\n'
        'print(\\\'[Stage2 / 09 eval preflight]\\\')\n'
        'print(\\\'  Checkpoint tar :\\\', CHECKPOINT_TAR)\n'
        'print(\\\'  Local checkpt  :\\\', CKPT)\n'
        'print(\\\'  Eval output    :\\\', os.path.join(DRIVE_TRAINING_RUNS, \\\'stage2_eval_summary.json\\\'))\n'
        'print(\\\'=\\\' * 60)\n'
    ))

    cells.append(_md('## 2. Rebuild the model and load weights'))
    cells.append(_code(
        'import torch, torch.nn as nn\n'
        'from lib.models.common import C3, Conv\n'
        'from stage2.fusion.lane_head import CurveLaneHead\n'
        '\n'
        'class TinyCSPBackbone(nn.Module):\n'
        '    def __init__(self):\n'
        '        super().__init__()\n'
        '        self.stem  = Conv(3, 32, 6, 2, 2)\n'
        '        self.down1 = Conv(32, 64, 3, 2);  self.c3_1 = C3(64, 64, 1)\n'
        '        self.down2 = Conv(64, 128, 3, 2); self.c3_2 = C3(128, 128, 2)\n'
        '        self.down3 = Conv(128, 256, 3, 2); self.c3_3 = C3(256, 256, 2)\n'
        '        self.down4 = Conv(256, 512, 3, 2); self.c3_4 = C3(512, 512, 1)\n'
        '    def forward(self, x):\n'
        '        x = self.stem(x); x = self.down1(x); x = self.c3_1(x)\n'
        '        x = self.down2(x); p3 = self.c3_2(x)\n'
        '        x = self.down3(p3); p4 = self.c3_3(x)\n'
        '        x = self.down4(p4); p5 = self.c3_4(x)\n'
        '        return [p3, p4, p5]\n'
        '\n'
        'device = torch.device(\\\'cuda\\\' if torch.cuda.is_available() else \\\'cpu\\\')\n'
        'state = torch.load(CKPT, map_location=device)\n'
        'fcfg  = state[\\\'config\\\']\n'
        'backbone  = TinyCSPBackbone().to(device); backbone.load_state_dict(state[\\\'backbone\\\'])\n'
        'lane_head = CurveLaneHead(\n'
        '    in_channels=fcfg[\\\'model\\\'][\\\'feature_channels\\\'],\n'
        '    embed_dim=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'embed_dim\\\'],\n'
        '    max_lanes=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'max_lanes\\\'],\n'
        '    num_points=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'num_points\\\'],\n'
        '    mask_size=tuple(fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'mask_size\\\']),\n'
        '    mask_aux=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'mask_aux\\\'],\n'
        '    num_lane_classes=fcfg[\\\'model\\\'][\\\'lane_head\\\'][\\\'num_lane_classes\\\'],\n'
        ').to(device); lane_head.load_state_dict(state[\\\'lane_head\\\'])\n'
        'backbone.eval(); lane_head.eval()\n'
        'print(\\\'Loaded checkpoint epoch=\\\', state.get(\\\'epoch\\\'), \\\' val_loss=\\\', state.get(\\\'val_loss\\\'))\n'
    ))

    cells.append(_md('## 3. Evaluate on BDD100K val'))
    cells.append(_code(
        '# Reuse the StageTwoCurveDataset from notebook 06.\n'
        'import cv2, numpy as np\n'
        'from torch.utils.data import Dataset, DataLoader\n'
        'from stage2.fusion.lane_targets import LaneLabelCache, soft_polyline_mask_numpy\n'
        '\n'
        'class StageTwoCurveDataset(Dataset):\n'
        '    def __init__(self, image_dir, label_dir, max_lanes=10, num_points=72,\n'
        '                 image_hw=(384, 640), aux_mask_size=(72, 128)):\n'
        '        self.image_dir = image_dir\n'
        '        self.cache = LaneLabelCache(label_dir, max_lanes=max_lanes, num_points=num_points)\n'
        '        self.image_hw = image_hw\n'
        '        self.aux_mask_size = aux_mask_size\n'
        '        self.names = [n for n in self.cache.keys() if os.path.exists(os.path.join(image_dir, n)) and self.cache.has_lanes(n)]\n'
        '    def __len__(self): return len(self.names)\n'
        '    def __getitem__(self, idx):\n'
        '        name = self.names[idx]\n'
        '        img = cv2.imread(os.path.join(self.image_dir, name)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n'
        '        img = cv2.resize(img, (self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_LINEAR)\n'
        '        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)\n'
        '        targets = self.cache.get(name)\n'
        '        mask = soft_polyline_mask_numpy(targets[\\\'points\\\'], targets[\\\'visibility\\\'], height=self.aux_mask_size[0], width=self.aux_mask_size[1])\n'
        '        return img_t, {\n'
        '            \\\'existence\\\': torch.from_numpy(targets[\\\'existence\\\']),\n'
        '            \\\'points\\\': torch.from_numpy(targets[\\\'points\\\']),\n'
        '            \\\'visibility\\\': torch.from_numpy(targets[\\\'visibility\\\']),\n'
        '            \\\'mask_target\\\': torch.from_numpy(mask).unsqueeze(0),\n'
        '        }\n'
        '\n'
        'def coll(batch):\n'
        '    imgs = torch.stack([b[0] for b in batch], 0)\n'
        '    keys = batch[0][1].keys()\n'
        '    return imgs, {k: torch.stack([b[1][k] for b in batch], 0) for k in keys}\n'
        '\n'
        'from lib.utils.drive_dataset import find_raw_bdd_root, resolve_bdd_images_100k_dir\n'
        'BDD_IMAGES = resolve_bdd_images_100k_dir(find_raw_bdd_root(DRIVE_ECOCAR), DRIVE_ECOCAR)\n'
        'val_ds = StageTwoCurveDataset(os.path.join(BDD_IMAGES, \\\'val\\\'), \\\'/content/bdd100k_raw/100k/val\\\')\n'
        'val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=coll)\n'
        '\n'
        'all_iou = []; all_mask_iou = []; all_existence_acc = []\n'
        'with torch.no_grad():\n'
        '    for img, tgt in val_loader:\n'
        '        img = img.to(device); tgt = {k: v.to(device) for k, v in tgt.items()}\n'
        '        feats = backbone(img); pred = lane_head(feats)\n'
        '        # 1) existence accuracy\n'
        '        existence_pred = (torch.sigmoid(pred[\\\'cls_logits\\\']) > 0.5).float()\n'
        '        all_existence_acc.append(((existence_pred == tgt[\\\'existence\\\']).float().mean()).item())\n'
        '        # 2) LineIoU per lane (rough proxy)\n'
        '        radius = 0.015\n'
        '        pa, ga = pred[\\\'coord_pred\\\'][..., 0], tgt[\\\'points\\\'][..., 0]\n'
        '        v = tgt[\\\'visibility\\\']\n'
        '        inter = (torch.minimum(pa+radius, ga+radius) - torch.maximum(pa-radius, ga-radius)).clamp(min=0)\n'
        '        union = (torch.maximum(pa+radius, ga+radius) - torch.minimum(pa-radius, ga-radius)).clamp(min=1e-6)\n'
        '        iou_per_lane = ((inter*v).sum(-1) / ((union*v).sum(-1) + 1e-6))\n'
        '        valid_lanes = tgt[\\\'existence\\\']\n'
        '        if valid_lanes.sum() > 0:\n'
        '            all_iou.append(((iou_per_lane * valid_lanes).sum() / valid_lanes.sum()).item())\n'
        '        # 3) mask IoU (binary @ 0.5)\n'
        '        if \\\'mask_logit\\\' in pred and \\\'mask_target\\\' in tgt:\n'
        '            pm = (torch.sigmoid(pred[\\\'mask_logit\\\']) > 0.5).float()\n'
        '            gm = (tgt[\\\'mask_target\\\'] > 0.5).float()\n'
        '            inter_m = (pm*gm).sum(); union_m = (pm + gm - pm*gm).sum().clamp(min=1)\n'
        '            all_mask_iou.append((inter_m / union_m).item())\n'
        '\n'
        'def _mean(xs):\n'
        '    return float(sum(xs) / max(len(xs), 1))\n'
        '\n'
        'eval_summary = {\n'
        '    \\\'lane_existence_acc\\\' : _mean(all_existence_acc),\n'
        '    \\\'lane_line_iou_proxy\\\' : _mean(all_iou),\n'
        '    \\\'lane_mask_iou\\\'       : _mean(all_mask_iou),\n'
        '}\n'
        'print(\\\'Eval summary:\\\', eval_summary)\n'
        '\n'
        'OUT_JSON = os.path.join(DRIVE_TRAINING_RUNS, \\\'stage2_eval_summary.json\\\')\n'
        'with open(OUT_JSON, \\\'w\\\') as fh:\n'
        '    import json as _json; _json.dump(eval_summary, fh, indent=2)\n'
        'print(\\\'Wrote\\\', OUT_JSON)\n'
    ))

    cells.append(_md('## 4. Per-stage timing'))
    cells.append(_code(
        'import time\n'
        'with torch.no_grad():\n'
        '    x = torch.randn(1, 3, 384, 640, device=device)\n'
        '    for _ in range(20): backbone(x); lane_head(backbone(x))   # warmup\n'
        '    if torch.cuda.is_available(): torch.cuda.synchronize()\n'
        '    t0 = time.perf_counter(); feats = backbone(x)\n'
        '    if torch.cuda.is_available(): torch.cuda.synchronize()\n'
        '    t1 = time.perf_counter(); pred = lane_head(feats)\n'
        '    if torch.cuda.is_available(): torch.cuda.synchronize()\n'
        '    t2 = time.perf_counter()\n'
        '    print(f\\\'backbone : {(t1-t0)*1000:.2f} ms\\\')\n'
        '    print(f\\\'lane head: {(t2-t1)*1000:.2f} ms\\\')\n'
        '    print(f\\\'total    : {(t2-t0)*1000:.2f} ms\\\')\n'
        '    print(f\\\'FPS      : {1.0/(t2-t0):.1f}\\\')\n'
        '    if torch.cuda.is_available():\n'
        '        print(f\\\'peak mem : {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB\\\')\n'
    ))

    _save(NB_DIR / '09_stage2_eval_video_profile_colab.ipynb', cells, 'Stage 2 / 09 eval + video profile')


def main() -> None:
    build_04()
    build_05()
    build_06()
    build_07()
    build_08()
    build_09()


if __name__ == '__main__':
    main()
