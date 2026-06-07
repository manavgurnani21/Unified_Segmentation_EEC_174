"""Generate NB61 (Exp2FFF), NB62 (Exp2GGG), NB63 (Exp2HHH)."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks'
TEMPLATE = NOTEBOOKS / 'stage2_notebook_40_exp2kk_anchor_asl_amp_joint.ipynb'


def _new_nb(template_path):
    nb = json.loads(template_path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return nb


def _smoke(cfg):
    return (
        "from pathlib import Path\nimport os, sys\n\n"
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
        "from pathlib import Path\nimport os, sys\n\n"
        f"CONFIG = '{cfg}'\n"
        "CURVE_TAR = '/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'\n"
        "CURVE_ROOT = '/content/bdd100k_clrkd_curve'\n\n"
        "DEBUG_MODE = False\n\n"
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
        "    PRINT_EVERY = 50\n\n"
        "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
        "WORK_DIR = f'/content/{run_stem}'\n"
        "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{run_stem}.tar'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{run_stem}_train.log')\n\n"
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
        "    cmd.extend(['--limit-train', str(LIMIT_TRAIN)])\n\n"
        "print('DEBUG_MODE:', DEBUG_MODE, flush=True)\n"
        "print('LIMIT_TRAIN:', LIMIT_TRAIN, flush=True)\n"
        "print('About to run:', ' '.join(cmd), flush=True)\n"
        "print('Output tar:', OUTPUT_TAR, flush=True)\n"
        "print('Visible log file:', LOG_FILE, flush=True)\n"
        "run_streaming(cmd, log_path=LOG_FILE)"
    )


def write_nb61():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 61 - Exp2FFF DN-DETR + lambda_det=0 (lane-only effective)\n"
        "\n"
        "**Isolate joint conflict from DAB-DETR instability.** NB59 hit val_lane_f1=0.623 at "
        "epoch 3 (project record) but the cls OSCILLATED: 0.458 ep1, 0.272 ep2, 0.623 ep3, "
        "0.277 ep4, 0.594 ep11, 0.222 ep15, 0.240 ep20. Hypothesis: the oscillation is caused "
        "by det's gradient pushing the shared backbone, which moves the DAB anchor parameters "
        "between stable configurations.\n"
        "\n"
        "Exp2FFF: set `lambda_det=0.0`, effectively training lane-only. Det loss still computed "
        "for logging but doesn't contribute to backward. If val_lane_f1 STABILIZES at 0.5+ "
        "without oscillation, joint conflict was the cause. If it still oscillates, DAB anchors "
        "are intrinsically unstable and we need a different stabilization strategy.\n"
        "\n"
        "Same head as NB59 (DN-DETR, K=64, DAB anchors, DN denoising) and same long head_warmup "
        "(10 ep) + bb_mult=0.02. Single config change: `lambda_det: 1.5 -> 0.0`."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 20 epochs, limit=3000."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp56_rmt_gca_query64_dn_lane_only_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp56_rmt_gca_query64_dn_lane_only_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2FFF\n"
        "\n"
        "Reference NB59: val_lane_f1 oscillates 0.22-0.62 across epochs, peak at ep3=0.623.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **val_lane_f1 STAYS >= 0.45 from epoch 5 onwards** -- no oscillation = joint "
        "conflict was the cause.\n"
        "- val_lane_best_f1 STAYS >= 0.50.\n"
        "- gap stays >= 0.15.\n"
        "- val_det may climb (we are not training it) but that's OK.\n"
        "\n"
        "If val_lane_f1 still oscillates: DAB anchors are intrinsically unstable; need "
        "Exp2HHH-style aggressive LR/weight-decay throttling."
    )
    out = NOTEBOOKS / 'stage2_notebook_61_exp2fff_query64_dn_lane_only_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb62():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 62 - Exp2GGG Anchor + cls_separate_path + VFL + full 70K + 12 epochs\n"
        "\n"
        "**Combine NB58's small cls win with NB60's geometry champion setup.** NB58 showed "
        "`cls_separate_path=True` gives gap=0.011 (vs 0.002 baseline) and val_lane_f1=0.099 "
        "(vs 0.000). Tiny improvement but real, on 3K data.\n"
        "\n"
        "Exp2GGG: keep `cls_separate_path=True` and scale up to NB60's recipe (full 70K, "
        "bb_throttle=0.01, 12 epochs, dynamic_k matching). Tests whether the disjoint cls "
        "feature pathway scales: maybe the small win on 3K becomes a meaningful win on 70K.\n"
        "\n"
        "Single diff vs NB60 (exp55): `cls_separate_path: false -> true`. Adds ~5% lane head "
        "params but no other change."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~2.5-3 hr wall-clock."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp57_rmt_gca_anchor_cls_sep_vfl_full_data_long12_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp57_rmt_gca_anchor_cls_sep_vfl_full_data_long12_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2GGG\n"
        "\n"
        "Reference NB60 (full + bb_throttle + 12 ep): matched_iou=0.554, val_lane_best_f1=0.119, "
        "gap=0.021, decoded_f1=0.050.\n"
        "Reference NB58 (3K + cls_sep + topk=3): matched_iou=0.374, val_lane_best_f1=0.147, "
        "gap=0.011, decoded_f1=0.047.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- val/matched_line_iou >= 0.55 (match NB60).\n"
        "- **val/lane_best_f1 >= 0.20** -- cls_separate_path at full data scale gives 2x the "
        "best_f1 of NB60.\n"
        "- pos_score - neg_score >= 0.05 (5x NB60's 0.021).\n"
        "- val/lane/decoded_f1 >= 0.10."
    )
    out = NOTEBOOKS / 'stage2_notebook_62_exp2ggg_anchor_cls_sep_vfl_full_data_long12_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb63():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 63 - Exp2HHH DN-DETR + ultra-stable training (lr=1e-4, wd=1e-3, clip=2)\n"
        "\n"
        "**Aggressive stabilization of the DN-DETR head.** NB59 confirmed the architecture can "
        "hit val_lane_f1=0.62 at epoch 3 but oscillates wildly between 0.22 and 0.62. The "
        "DAB anchor parameters drift between stable configurations due to high effective LR.\n"
        "\n"
        "Exp2HHH: brute-force stabilization via training-side hyperparameters:\n"
        "- `lr0: 0.0002 -> 0.0001` (halve main LR)\n"
        "- `backbone_lr_mult: 0.02 -> 0.005` (4x slower backbone, 200x slower than baseline)\n"
        "- `weight_decay: 0.0005 -> 0.001` (2x stronger weight regularization)\n"
        "- `grad_clip_norm: 5.0 -> 2.0` (tighter gradient clipping)\n"
        "- `head_warmup until_epoch: 10 -> 12` (longer head-only training)\n"
        "- `lambda_det: 1.0 -> 1.5` (slight det boost to avoid the late-epoch det collapse "
        "NB59 showed)\n"
        "- `use_uncertainty: true -> false`\n"
        "\n"
        "Hypothesis: DAB anchors move slowly enough to stay near optimum. Should give STABLE "
        "val_lane_f1 ≥ 0.50 from epoch 10 onwards. Wall-clock similar to NB59 (~35 min)."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 20 epochs limit=3000."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp58_rmt_gca_query64_dn_vfl_stable_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp58_rmt_gca_query64_dn_vfl_stable_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2HHH\n"
        "\n"
        "Reference NB59 (DN-DETR + 10ep warmup): val_lane_f1 oscillates 0.22-0.62, peak ep3=0.623.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **val_lane_f1 >= 0.40 SUSTAINED from epoch 10 onwards** -- no oscillation = "
        "training-side stabilization worked.\n"
        "- val_lane_best_f1 >= 0.50.\n"
        "- gap >= 0.15 at epoch 20.\n"
        "- matched_iou >= 0.20 (geometry still trained though slower).\n"
        "- val_det <= 2.5 (det not crashed).\n"
        "\n"
        "If sustained val_lane_f1 >= 0.40 + matched_iou >= 0.20: we have a DEPLOYABLE query "
        "head model with the cls breakthrough preserved. Combine with anchor's curves at "
        "inference (anchor coord_pred + query cls scoring) for the final fusion model.\n"
        "\n"
        "If still oscillates: need code-level intervention (EMA model averaging or static "
        "anchor positions)."
    )
    out = NOTEBOOKS / 'stage2_notebook_63_exp2hhh_query64_dn_vfl_stable_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb61, write_nb62, write_nb63):
        out = fn()
        print('wrote:', out)
