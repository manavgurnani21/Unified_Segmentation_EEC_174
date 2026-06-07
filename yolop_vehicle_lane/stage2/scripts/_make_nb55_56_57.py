"""Generate NB55 (Exp2ZZ), NB56 (Exp2AAA), NB57 (Exp2BBB)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks'
TEMPLATE = NOTEBOOKS / 'stage2_notebook_40_exp2kk_anchor_asl_amp_joint.ipynb'


def _new_nb(template_path: Path):
    nb = json.loads(template_path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return nb


def _smoke(cfg):
    return (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        f"CONFIG = '{cfg}'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{Path(CONFIG).stem}_smoke.log')\n"
        "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
    )


def _train(cfg, run_tag, epochs, limit_train, limit_val=1000):
    if limit_train is None:
        limit_block = "LIMIT_TRAIN = None\n"
    else:
        limit_block = f"LIMIT_TRAIN = {limit_train}\n"
    return (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        f"CONFIG = '{cfg}'\n"
        "CURVE_TAR = '/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'\n"
        "CURVE_ROOT = '/content/bdd100k_clrkd_curve'\n"
        "\n"
        "DEBUG_MODE = False\n"
        "\n"
        "if DEBUG_MODE:\n"
        "    RUN_TAG = 'debug'\n"
        "    EPOCHS = 2\n"
        "    BATCH_SIZE = 4\n"
        "    LIMIT_TRAIN = 512\n"
        "    LIMIT_VAL = 256\n"
        "    PRINT_EVERY = 5\n"
        "else:\n"
        f"    RUN_TAG = '{run_tag}'\n"
        f"    EPOCHS = {epochs}\n"
        "    BATCH_SIZE = 8\n"
        f"    {limit_block}"
        f"    LIMIT_VAL = {limit_val}\n"
        "    PRINT_EVERY = 50\n"
        "\n"
        "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
        "WORK_DIR = f'/content/{run_stem}'\n"
        "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{run_stem}.tar'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{run_stem}_train.log')\n"
        "\n"
        "cmd = [\n"
        "    sys.executable, '-u', 'stage2/scripts/train_joint_model_experiment.py',\n"
        "    '--config', CONFIG,\n"
        "    '--curve-tar', CURVE_TAR,\n"
        "    '--curve-root', CURVE_ROOT,\n"
        "    '--work-dir', WORK_DIR,\n"
        "    '--output-tar', OUTPUT_TAR,\n"
        "    '--epochs', str(EPOCHS),\n"
        "    '--batch-size', str(BATCH_SIZE),\n"
        "    '--limit-val', str(LIMIT_VAL),\n"
        "    '--force-extract',\n"
        "    '--print-every', str(PRINT_EVERY),\n"
        "]\n"
        "if LIMIT_TRAIN is not None:\n"
        "    cmd.extend(['--limit-train', str(LIMIT_TRAIN)])\n"
        "\n"
        "print('DEBUG_MODE:', DEBUG_MODE, flush=True)\n"
        "print('LIMIT_TRAIN:', LIMIT_TRAIN, flush=True)\n"
        "print('About to run:', ' '.join(cmd), flush=True)\n"
        "print('Output tar:', OUTPUT_TAR, flush=True)\n"
        "print('Visible log file:', LOG_FILE, flush=True)\n"
        "run_streaming(cmd, log_path=LOG_FILE)"
    )


def write_nb55():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 55 - Exp2ZZ Anchor + topk-fixed K=3 + VFL\n"
        "\n"
        "**Density-tuned topk_fixed.** NB52 (Exp2WW, K=8) crashed geometry to matched_iou=0.256 vs NB48's "
        "0.544. With 8 priors per GT all chasing the same curve, gradient spread thin -- no prior got a tight "
        "fit. The matcher fix was on the right axis (pos-neg gap doubled from 0.001 -> 0.003) just with too "
        "many positives.\n"
        "\n"
        "Exp2ZZ: K=3 instead. 3 priors per GT * 5 GT = ~12-15 positives/image, close to dynamic-k's empirical "
        "8-12. Stable labels (Exp2WW's win), less geometric dilution.\n"
        "\n"
        "Single config diff vs NB52 (Exp2WW exp47): `topk_fixed_per_gt: 8 -> 3`. All else identical."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n1. `DEBUG_MODE=True` smoke.\n2. `DEBUG_MODE=False` 20 epochs limit=3000."
    )
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp50_rmt_gca_anchor_topk3_vfl_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp50_rmt_gca_anchor_topk3_vfl_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2ZZ\n"
        "\n"
        "Reference NB52 (K=8): matched_iou=0.256, gap=0.003, decoded_f1=0.043.\n"
        "Reference NB48 (dynamic_k full data): matched_iou=0.544, decoded_f1=0.050.\n"
        "Reference NB54 (K=8 + full data): matched_iou=0.289, decoded_f1=0.062.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- `val/matched_line_iou >= 0.40` (recover most of NB48's geometry).\n"
        "- `pos_score - neg_score >= 0.02` (stable labels still help cls).\n"
        "- `val/lane/decoded_f1 >= 0.07` (beat NB54's record of 0.062).\n"
        "- `val/lane_best_f1 >= 0.20`."
    )
    out = NOTEBOOKS / 'stage2_notebook_55_exp2zz_anchor_topk3_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb56():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 56 - Exp2AAA Query K=64 + DAB + DN + VFL + 30 epochs\n"
        "\n"
        "**Let the DN-DETR queries converge.** NB53 (Exp2XX) showed the highest cls discrimination ever "
        "measured (val_lane_best_f1=0.476 at epoch 7, gap=0.157 at epoch 1) but matched_iou stayed at 0.195 "
        "and val_det collapsed to 3.35 at epoch 7+. DN-DETR papers (Li et al. 2022) typically need 50+ "
        "epochs at standard batch sizes for queries to ground themselves -- 20 epochs at our setup was "
        "premature.\n"
        "\n"
        "Exp2AAA: same head as NB53, but 30 epochs at limit=3000 with cosine LR, lambda_det=1.5 and "
        "use_uncertainty=False to prevent the late-training det collapse NB53 showed.\n"
        "\n"
        "Diffs vs NB53 (Exp2XX exp48):\n"
        "- `end_epoch: 20 -> 30`\n"
        "- `lambda_det: 1.0 -> 1.5`, `use_uncertainty: true -> false`\n"
        "- `warmup_epochs: 2 -> 3`"
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n1. `DEBUG_MODE=True` smoke.\n2. `DEBUG_MODE=False` 30 epochs limit=3000. ~50 min."
    )
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp51_rmt_gca_query64_dn_vfl_long30_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp51_rmt_gca_query64_dn_vfl_long30_joint.yaml',
        run_tag='short30', epochs=30, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2AAA\n"
        "\n"
        "Reference NB53: matched_iou=0.195 (epoch 20), val_lane_best_f1=0.476 (epoch 7, peak), val_det "
        "crashed to 3.3 at epoch 7+.\n"
        "\n"
        "Pass criteria at epoch 30:\n"
        "- `val/matched_line_iou >= 0.30` -- 1.5x NB53; longer training should ground queries geometrically.\n"
        "- `val/lane_best_f1 >= 0.40` -- maintain NB53 peak.\n"
        "- `val/lane/decoded_f1 >= 0.05` -- 2x NB53.\n"
        "- `val_det <= 2.5` AT EPOCH 30 -- explicit det rescue should prevent the NB53-style late collapse."
    )
    out = NOTEBOOKS / 'stage2_notebook_56_exp2aaa_query64_dn_vfl_long30_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb57():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 57 - Exp2BBB Anchor + VFL + full 70K + backbone-LR throttle (joint conflict fix)\n"
        "\n"
        "**The diagnosed joint-conflict fix.** NB48/51/54 all hit val_map50 = 0 at full data despite "
        "lambda_det re-weighting (NB48: 1.0, NB51: 2.0, NB54: 3.0). Re-weighting the head losses doesn't fix "
        "the underlying conflict because BOTH losses backpropagate through the same shared backbone, and at "
        "full data the lane gradient norm dominates.\n"
        "\n"
        "Diagnosis: the issue is in BACKBONE updates, not head weighting. With `backbone_lr_mult=0.1` "
        "(default), the backbone moves significantly each step under both head gradients. Lane wins the "
        "tug-of-war at full data scale.\n"
        "\n"
        "Fix: `backbone_lr_mult: 0.1 -> 0.01`. Backbone effectively quasi-frozen (still trains, just 10x "
        "slower than heads). Both tasks adapt to the slowly-moving backbone rather than fighting over it. "
        "This is the standard DETR-style 'lr_backbone=1e-6' recipe.\n"
        "\n"
        "Diffs vs NB54 (Exp2YY exp49):\n"
        "- `backbone_lr_mult: 0.1 -> 0.01` (the only critical change)\n"
        "- `lambda_det: 3.0 -> 1.5` (gentler since we expect the backbone fix to fix det)\n"
        "- `lambda_lane: 0.5 -> 1.0` (don't suppress lane)\n"
        "- matcher: `topk_fixed -> dynamic_k` (preserve NB48's geometry champion)"
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n1. `DEBUG_MODE=True` smoke.\n2. `DEBUG_MODE=False` 6 epochs full data. ~60-80 min."
    )
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp52_rmt_gca_anchor_vfl_full_data_bb_throttle_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp52_rmt_gca_anchor_vfl_full_data_bb_throttle_joint.yaml',
        run_tag='full6', epochs=6, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2BBB\n"
        "\n"
        "Reference NB48: matched_iou=0.544, decoded_f1=0.050, val_det=3.16, val_map50=0.\n"
        "Reference NB54: matched_iou=0.289, decoded_f1=0.062, val_det=3.10, val_map50=0.\n"
        "\n"
        "Pass criteria at epoch 6:\n"
        "- **`val_det <= 2.5` and `val/det/map50 >= 0.005`** -- the smoking gun. If backbone throttling "
        "fixes det, this hits.\n"
        "- `val/matched_line_iou >= 0.45` -- accept a small regression from NB48's 0.544 (backbone moves "
        "less, so lane head trains slightly less geometry).\n"
        "- `val/lane/decoded_f1 >= 0.04` -- maintain NB48 level.\n"
        "- `train/grad_cosine_epoch_mean >= 0` -- diagnostic confirming the conflict is reduced."
    )
    out = NOTEBOOKS / 'stage2_notebook_57_exp2bbb_anchor_vfl_full_data_bb_throttle_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb55, write_nb56, write_nb57):
        out = fn()
        print('wrote:', out)
