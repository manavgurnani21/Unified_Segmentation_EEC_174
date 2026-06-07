"""Generate NB73 (speed test), NB74 (CULane pretrain), NB75 (joint from CULane init)."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks'
TEMPLATE = NOTEBOOKS / 'stage2_notebook_40_exp2kk_anchor_asl_amp_joint.ipynb'


def _new_nb():
    nb = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    for c in nb['cells']:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    return nb


def _smoke(cfg):
    return (
        "from pathlib import Path\n"
        "import os, sys\n\n"
        f"CONFIG = '{cfg}'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{Path(CONFIG).stem}_smoke.log')\n"
        "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
    )


def write_nb73():
    """NB73 — speed test on existing best lane recipe (NB62) with batch=32 + torch.compile."""
    nb = _new_nb()
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 73 - Speed test: NB62 recipe + batch=32 + workers=6 + torch.compile\n"
        "\n"
        "**Priority 1 of the speed/pretrained/CULane plan.** NB70 took 10h, NB71 took 8h at batch=8, "
        "workers=2, no torch.compile. RTX Pro 6000 was 5% utilized. This run tests the same lane recipe "
        "(NB62 = anchor + cls_separate_path + VFL + bb_throttle) with:\n"
        "\n"
        "- `batch_size: 8 -> 32` (4x throughput per step)\n"
        "- `workers: 2 -> 6` (less dataloader idle)\n"
        "- `lr0: 2e-4 -> 4e-4` (sqrt(4)x scaling for batch increase)\n"
        "- `torch.compile(reduce-overhead)` (+20-40% on transformer)\n"
        "- `persistent_workers=True` + `prefetch_factor=4`\n"
        "\n"
        "Expected: full 70K / 32 = 2188 iters/epoch (vs 8750), ~3x speedup per epoch. 12 epochs should "
        "drop from 14h (NB62) to ~2-3h.\n"
        "\n"
        "If metrics match NB62 (matched_iou ~0.55, decoded_f1 ~0.06): speed flags are safe defaults for "
        "all future runs and we add them to every yaml.\n"
        "If metrics regress: LR scaling was wrong, revert to lr0=2e-4 or use gradient accumulation."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~2-3 hr (was ~14 hr at batch=8)."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp71_rmt_gca_anchor_cls_sep_vfl_speed_batch32_joint.yaml')
    nb['cells'][4]['source'] = (
        "from pathlib import Path\n"
        "import os, sys\n\n"
        "CONFIG = 'stage2/configs/exp71_rmt_gca_anchor_cls_sep_vfl_speed_batch32_joint.yaml'\n"
        "CURVE_TAR = '/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'\n"
        "CURVE_ROOT = '/content/bdd100k_clrkd_curve'\n\n"
        "DEBUG_MODE = False\n\n"
        "if DEBUG_MODE:\n"
        "    RUN_TAG = 'debug'\n    EPOCHS = 2\n    BATCH_SIZE = 8\n    LIMIT_TRAIN = 512\n    LIMIT_VAL = 256\n    PRINT_EVERY = 5\n"
        "else:\n    RUN_TAG = 'full12_speed'\n    EPOCHS = 12\n    BATCH_SIZE = 32\n    LIMIT_TRAIN = None\n    LIMIT_VAL = 1000\n    PRINT_EVERY = 50\n\n"
        "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
        "WORK_DIR = f'/content/{run_stem}'\n"
        "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{run_stem}.tar'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{run_stem}_train.log')\n\n"
        "cmd = [\n"
        "    sys.executable, '-u', 'stage2/scripts/train_joint_model_experiment.py',\n"
        "    '--config', CONFIG,\n    '--curve-tar', CURVE_TAR,\n    '--curve-root', CURVE_ROOT,\n"
        "    '--work-dir', WORK_DIR,\n    '--output-tar', OUTPUT_TAR,\n"
        "    '--epochs', str(EPOCHS),\n    '--batch-size', str(BATCH_SIZE),\n"
        "    '--limit-val', str(LIMIT_VAL),\n    '--force-extract',\n    '--print-every', str(PRINT_EVERY),\n"
        "    # SPEED FLAGS:\n"
        "    '--workers', '6',\n"
        "    '--prefetch-factor', '4',\n"
        "    '--torch-compile',\n"
        "]\n"
        "if LIMIT_TRAIN is not None:\n    cmd.extend(['--limit-train', str(LIMIT_TRAIN)])\n\n"
        "print('DEBUG_MODE:', DEBUG_MODE, flush=True)\nprint('LIMIT_TRAIN:', LIMIT_TRAIN, flush=True)\n"
        "print('BATCH_SIZE:', BATCH_SIZE, flush=True)\n"
        "print('About to run:', ' '.join(cmd), flush=True)\nprint('Output tar:', OUTPUT_TAR, flush=True)\n"
        "print('Visible log file:', LOG_FILE, flush=True)\nrun_streaming(cmd, log_path=LOG_FILE)"
    )
    nb['cells'][5]['source'] = (
        "## What to watch in NB73 (speed test)\n"
        "\n"
        "Reference NB62 (batch=8, 12 ep full, ~14 hr): matched_iou=0.55, decoded_f1=0.06, gap=0.024, val_map50=0.\n"
        "\n"
        "Pass criteria:\n"
        "- **Wall-clock <= 4 hr** (5x speedup vs NB62).\n"
        "- `val/matched_line_iou >= 0.50` -- preserve NB62 geometry within noise.\n"
        "- `val/lane/decoded_f1 >= 0.05` -- preserve NB62 cls.\n"
        "- `[loader]` log line shows workers=6 persistent=True prefetch=4.\n"
        "- `[speed] torch.compile(...)` log line at start.\n"
        "- `peak_mem_mb` ~ 30-40 GB (vs NB62's 10 GB).\n"
        "\n"
        "If pass: these flags become default for ALL future configs.\n"
        "If matched_iou drops > 0.05: LR scaling was wrong. Switch to gradient accumulation "
        "(--batch-size 8 --grad-accum-steps 4) to simulate batch=32 with NB62's LR."
    )
    out = NOTEBOOKS / 'stage2_notebook_73_speed_test_batch32_compile_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb74():
    """NB74 — CULane lane-only pretraining."""
    nb = _new_nb()
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 74 - CULane lane-only pretraining\n"
        "\n"
        "**Priority 3.** Pretrain the joint model's BACKBONE on CULane (88K images, lane-only, no det "
        "conflict, 8 epochs) so it has good lane-aware features BEFORE we joint-train on BDD100K.\n"
        "\n"
        "Pipeline:\n"
        "1. **Extract CULane archives** from `MyDrive/EcoCAR/downloads/CULane/` to `/content/CULane/`.\n"
        "   - list.tar.gz (split files)\n"
        "   - annotations_new.tar.gz (.lines.txt labels)\n"
        "   - driver_<X>_<Y>frame.tar.gz (images; we extract ALL six)\n"
        "2. **Train lane-only** (lambda_det=0, --allow-empty-det-labels) for 8 epochs at batch=32.\n"
        "3. **Export backbone_pretrained.pt** -- the training script now saves a backbone-only checkpoint\n"
        "   automatically. We'll use this in NB75 as the joint-training init.\n"
        "\n"
        "Note: CULane has different image dimensions (590x1640 vs BDD 720x1280). Both are resized to "
        "384x640 by the dataloader so this is a noop at the network level. CULane has max_lanes=4 (vs "
        "BDD max_lanes=10) which means the lane HEAD outputs 192 priors / 4 GTs vs 192 priors / 10 GTs "
        "during pretrain. The BACKBONE features are what we keep -- we don't transfer the lane head."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "1. Extract CULane archives (~40 GB; may take 30-60 min).\n"
        "2. Smoke.\n"
        "3. 8 epochs CULane train (~2-3 hr at batch=32 with speed flags).\n"
        "4. Backbone checkpoint is auto-saved as `backbone_pretrained.pt` in the work dir."
    )
    nb['cells'][3]['source'] = (
        "# Step 1: extract CULane archives.\n"
        "from pathlib import Path\n"
        "import os, sys, subprocess\n"
        "\n"
        "CULANE_SRC = '/content/drive/MyDrive/EcoCAR/downloads/CULane'\n"
        "CULANE_DST = '/content/CULane'\n"
        "\n"
        "cmd = [sys.executable, '-u', 'stage2/scripts/prepare_culane_dataset.py',\n"
        "       '--src', CULANE_SRC,\n"
        "       '--dest', CULANE_DST]\n"
        "LOG_FILE = os.path.join(LOG_DIR, 'culane_prepare.log')\n"
        "print('Extracting CULane archives...')\n"
        "run_streaming(cmd, log_path=LOG_FILE)"
    )
    nb['cells'][4]['source'] = (
        "# Step 2: train lane-only on CULane for 8 epochs.\n"
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "CONFIG = 'stage2/configs/exp69_rmt_gca_culane_lane_only_pretrain.yaml'\n"
        "\n"
        "DEBUG_MODE = False\n"
        "if DEBUG_MODE:\n"
        "    RUN_TAG = 'debug'\n    EPOCHS = 2\n    BATCH_SIZE = 8\n    LIMIT_TRAIN = 512\n    LIMIT_VAL = 256\n    PRINT_EVERY = 5\n"
        "else:\n    RUN_TAG = 'culane8'\n    EPOCHS = 8\n    BATCH_SIZE = 32\n    LIMIT_TRAIN = None\n    LIMIT_VAL = 1000\n    PRINT_EVERY = 100\n\n"
        "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
        "WORK_DIR = f'/content/{run_stem}'\n"
        "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{run_stem}.tar'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{run_stem}_train.log')\n\n"
        "cmd = [\n"
        "    sys.executable, '-u', 'stage2/scripts/train_joint_model_experiment.py',\n"
        "    '--config', CONFIG,\n"
        "    # No curve-tar; the CULane data is already extracted to CULANE_DST.\n"
        "    '--curve-root', '/content/CULane',\n"
        "    '--work-dir', WORK_DIR,\n"
        "    '--output-tar', OUTPUT_TAR,\n"
        "    '--epochs', str(EPOCHS),\n    '--batch-size', str(BATCH_SIZE),\n"
        "    '--limit-val', str(LIMIT_VAL),\n    '--print-every', str(PRINT_EVERY),\n"
        "    # CULane has no detection labels.\n"
        "    '--allow-empty-det-labels',\n"
        "    # Speed flags.\n"
        "    '--workers', '6',\n    '--prefetch-factor', '4',\n    '--torch-compile',\n"
        "]\n"
        "if LIMIT_TRAIN is not None:\n    cmd.extend(['--limit-train', str(LIMIT_TRAIN)])\n\n"
        "print('Training CULane lane-only pretrain. Backbone checkpoint will be saved at:')\n"
        "print(f'  {WORK_DIR}/backbone_pretrained.pt')\n"
        "print('and copied into the output tar.')\n"
        "print('About to run:', ' '.join(cmd), flush=True)\n"
        "run_streaming(cmd, log_path=LOG_FILE)"
    )
    nb['cells'][5]['source'] = (
        "## What to watch in NB74 (CULane pretrain)\n"
        "\n"
        "There is no published reference for our specific config on CULane, but rough expectations:\n"
        "- `val/matched_line_iou >= 0.50` by epoch 8 (CULane is a single-task lane benchmark).\n"
        "- `val/lane/decoded_f1 >= 0.30` (CULane is easier than BDD for lane: cleaner labels, fewer crossings).\n"
        "- `val/lane/decoded_oracle_f1 >= 0.60`.\n"
        "- Wall-clock 2-4 hr.\n"
        "- `backbone_pretrained.pt` exists in the output tar and is 30-100 MB.\n"
        "\n"
        "If decoded_f1 stays < 0.10 on CULane: our lane head's cls problem is intrinsic to the head itself "
        "(not BDD-specific), and pretraining won't help. Pivot to swapping the lane head architecture.\n"
        "\n"
        "If decoded_f1 hits >= 0.30 on CULane: the head works when not fighting det. The BDD joint problem "
        "is partly capacity / joint-conflict / data-mismatch. Use this backbone in NB75 for the joint run."
    )
    out = NOTEBOOKS / 'stage2_notebook_74_culane_lane_only_pretrain.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb75():
    """NB75 — joint BDD training initialized from NB74's backbone."""
    nb = _new_nb()
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 75 - Joint BDD training initialized from NB74's CULane-pretrained backbone\n"
        "\n"
        "**Priorities 2 + 3 combined.** NB74 produced a CULane-pretrained `backbone_pretrained.pt`. This "
        "notebook loads it as init for a joint BDD100K run with the speed flags from NB73.\n"
        "\n"
        "Why this matters: across 40+ experiments the backbone was always random-init. The reason mAP50 stuck "
        "near 0 at full data and matched_iou plateaued around 0.55 is partly that the backbone is starting "
        "from noise on EVERY run. A CULane-trained backbone should give:\n"
        "- Faster joint convergence (det converges to mAP > 0 in 8-12 epochs instead of needing 30+)\n"
        "- Higher lane ceiling (matched_iou > 0.60 instead of ~0.55)\n"
        "- Possibly cracks the cls collapse (cls features start from a lane-aware basin)"
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "1. Confirm NB74 produced `backbone_pretrained.pt` in Drive.\n"
        "2. Smoke.\n"
        "3. 8 epochs full BDD with the CULane init + speed flags + torch.compile. ~2 hr."
    )
    nb['cells'][3]['source'] = (
        "# Extract NB74's backbone-only checkpoint from its output tar.\n"
        "from pathlib import Path\n"
        "import os, sys, subprocess\n"
        "\n"
        "NB74_TAR = '/content/drive/MyDrive/EcoCAR/training_runs/exp69_rmt_gca_culane_lane_only_pretrain_culane8.tar'\n"
        "PRETRAIN_DIR = '/content/exp69_extracted'\n"
        "PRETRAIN_BACKBONE = f'{PRETRAIN_DIR}/backbone_pretrained.pt'\n"
        "\n"
        "Path(PRETRAIN_DIR).mkdir(parents=True, exist_ok=True)\n"
        "print(f'Extracting CULane-pretrained backbone from {NB74_TAR}...')\n"
        "subprocess.check_call(['tar', '-xf', NB74_TAR, '-C', PRETRAIN_DIR])\n"
        "assert Path(PRETRAIN_BACKBONE).exists(), f'{PRETRAIN_BACKBONE} not in tar -- did NB74 finish?'\n"
        "print(f'Pretrained backbone ready at {PRETRAIN_BACKBONE}')\n"
        "subprocess.check_call(['ls', '-lh', PRETRAIN_BACKBONE])"
    )
    nb['cells'][4]['source'] = (
        "# Joint BDD training with CULane-pretrained backbone + speed flags.\n"
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "CONFIG = 'stage2/configs/exp70_rmt_gca_culane_init_anchor_cls_sep_vfl_joint.yaml'\n"
        "CURVE_TAR = '/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'\n"
        "CURVE_ROOT = '/content/bdd100k_clrkd_curve'\n"
        "PRETRAIN_BACKBONE = '/content/exp69_extracted/backbone_pretrained.pt'\n"
        "\n"
        "DEBUG_MODE = False\n"
        "if DEBUG_MODE:\n"
        "    RUN_TAG = 'debug'\n    EPOCHS = 2\n    BATCH_SIZE = 8\n    LIMIT_TRAIN = 512\n    LIMIT_VAL = 256\n    PRINT_EVERY = 5\n"
        "else:\n    RUN_TAG = 'full8_culane_init'\n    EPOCHS = 8\n    BATCH_SIZE = 32\n    LIMIT_TRAIN = None\n    LIMIT_VAL = 1000\n    PRINT_EVERY = 50\n\n"
        "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
        "WORK_DIR = f'/content/{run_stem}'\n"
        "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{run_stem}.tar'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{run_stem}_train.log')\n\n"
        "cmd = [\n"
        "    sys.executable, '-u', 'stage2/scripts/train_joint_model_experiment.py',\n"
        "    '--config', CONFIG,\n    '--curve-tar', CURVE_TAR,\n    '--curve-root', CURVE_ROOT,\n"
        "    '--work-dir', WORK_DIR,\n    '--output-tar', OUTPUT_TAR,\n"
        "    '--epochs', str(EPOCHS),\n    '--batch-size', str(BATCH_SIZE),\n"
        "    '--limit-val', str(LIMIT_VAL),\n    '--force-extract',\n    '--print-every', str(PRINT_EVERY),\n"
        "    # PRETRAINED INIT:\n"
        "    '--pretrained-backbone', PRETRAIN_BACKBONE,\n"
        "    # SPEED FLAGS:\n"
        "    '--workers', '6',\n    '--prefetch-factor', '4',\n    '--torch-compile',\n"
        "]\n"
        "if LIMIT_TRAIN is not None:\n    cmd.extend(['--limit-train', str(LIMIT_TRAIN)])\n\n"
        "print('DEBUG_MODE:', DEBUG_MODE, 'BATCH_SIZE:', BATCH_SIZE, flush=True)\n"
        "print('Using pretrained backbone:', PRETRAIN_BACKBONE, flush=True)\n"
        "print('About to run:', ' '.join(cmd), flush=True)\n"
        "run_streaming(cmd, log_path=LOG_FILE)"
    )
    nb['cells'][5]['source'] = (
        "## What to watch in NB75 (CULane-init joint training)\n"
        "\n"
        "Reference NB62 (random-init backbone, 12 ep full): matched_iou=0.55, decoded_f1=0.06, val_map50=0.\n"
        "Reference NB45 (random-init backbone, 3K data, 30 ep, width=1.0): val_map50=0.011 -- our best det.\n"
        "\n"
        "Pass criteria at epoch 8 (40% fewer epochs than NB62 but with pretrained init):\n"
        "- **`val/matched_line_iou >= 0.60`** -- the CULane-trained backbone has lane-aware features.\n"
        "- **`val/lane/decoded_f1 >= 0.10`** -- 1.7x NB62's 0.06; the lane head's cls problem partially "
        "fixed by better initial backbone features.\n"
        "- **`val_map50 >= 0.01`** -- det actually starts converging because random-init handicap is gone.\n"
        "- `[pretrained] loaded N/M tensors` log line shows reasonable load coverage (50%+).\n"
        "- Wall-clock 1-2 hr (vs NB62's 14 hr).\n"
        "\n"
        "If matched_iou >= 0.60 AND val_map50 >= 0.01: **the pretrained-init is the single biggest lift** "
        "we've had in 75 experiments. Stage 3 deployment uses this checkpoint.\n"
        "\n"
        "If pretrained loaded < 30% of tensors: the loader's name-matching is too strict. Check the log "
        "for the first few source/target keys and adjust `_normalize_keys` in pretrained_loader.py.\n"
        "\n"
        "If metrics don't improve over NB62: CULane pretrain was too short OR architecture mismatch. "
        "Retry with end_epoch=16 on NB74."
    )
    out = NOTEBOOKS / 'stage2_notebook_75_culane_init_anchor_cls_sep_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb73, write_nb74, write_nb75):
        out = fn()
        print('wrote:', out)
