"""Generate NB67 (Exp2LLL), NB68 (Exp2MMM), NB69 (Exp2NNN)."""
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


def write_nb67():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 67 - Exp2LLL Anchor + cls_sep + IoU-priority matching (tempered) + 14ep\n"
        "\n"
        "**Re-run NB66 with tempered IoU priority + longer training.** NB66 (exp61) failed mid-run with "
        "`tar: Transport endpoint is not connected` — Drive mount dropped. This is a fresh retry with "
        "slightly less aggressive IoU priority and longer training.\n"
        "\n"
        "NB62 used `cost_iou=2.0, cost_point=5.0, w_iou=2.0`. NB66 (failed) tried `cost_iou=5.0, "
        "cost_point=2.0, w_iou=3.0` -- full reversal. Exp2LLL: midpoint at `cost_iou=4.0, "
        "cost_point=3.0, w_iou=2.5`. Plus 14 epochs (vs NB62's 12). Tests whether moderate IoU-priority "
        "matching helps while preserving NB62's coordinate-distance signal.\n"
        "\n"
        "Diffs vs NB62 (exp57):\n"
        "- `match_cost_iou: 2.0 -> 4.0`\n"
        "- `match_cost_point: 5.0 -> 3.0`\n"
        "- `w_iou: 2.0 -> 2.5`\n"
        "- `end_epoch: 12 -> 14`"
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 14 epochs full 70K. ~3-3.5 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp62_rmt_gca_anchor_cls_sep_vfl_iou_match_long14_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp62_rmt_gca_anchor_cls_sep_vfl_iou_match_long14_joint.yaml',
        run_tag='full14', epochs=14, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2LLL\n"
        "\n"
        "Reference NB62 (cost_iou=2, cost_point=5): matched_iou=0.550, decoded_f1=0.073, gap=0.045.\n"
        "\n"
        "Pass criteria at epoch 14:\n"
        "- val/matched_line_iou >= 0.55 (preserved or improved by tempered IoU-priority).\n"
        "- val/lane/decoded_f1 >= 0.08 (beat NB62 by 10 %).\n"
        "- pos-neg gap >= 0.05.\n"
        "- val/lane_f1 >= 0.13."
    )
    out = NOTEBOOKS / 'stage2_notebook_67_exp2lll_anchor_cls_sep_vfl_iou_match_long14_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb68():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 68 - Exp2MMM Anchor + cls_sep + K=4 topk_fixed + full 70K + 12ep\n"
        "\n"
        "**The geometry/cls balance test.** Across recent experiments:\n"
        "- K=8 topk_fixed (NB52/54): matched_iou=0.26-0.29, val_lane_f1=0.07 (geometry crashed)\n"
        "- K=3 topk_fixed (NB65): matched_iou=0.40, val_lane_f1=0.169 (cls won but geometry lost)\n"
        "- dynamic_k (NB62): matched_iou=0.55, val_lane_f1=0.118 (geometry won but cls weaker)\n"
        "\n"
        "K=4 topk_fixed should be the middle ground. Stable labels (topk_fixed's win) with closer-to-"
        "dynamic_k density. Goal: preserve more of NB62's geometry (matched_iou ~ 0.50) while keeping "
        "NB65's cls gain (val_lane_f1 > 0.15).\n"
        "\n"
        "Single diff vs NB62 (exp57): `lane_assigner: dynamic_k -> topk_fixed`, `topk_fixed_per_gt: 4`."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~2.5-3 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp63_rmt_gca_anchor_cls_sep_topk4_vfl_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp63_rmt_gca_anchor_cls_sep_topk4_vfl_full_data_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2MMM\n"
        "\n"
        "Reference NB62 (dynamic_k + cls_sep): matched_iou=0.550, decoded_f1=0.073, val_lane_f1=0.118.\n"
        "Reference NB65 (topk_fixed K=3 + cls_sep): matched_iou=0.404, decoded_f1=0.061, val_lane_f1=0.169.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- **val/matched_line_iou >= 0.45** (recovery from K=3's geometry crash)\n"
        "- **val/lane_f1 >= 0.13** (above NB62, below NB65 acceptable)\n"
        "- val/lane/decoded_f1 >= 0.08 (beat both NB62 and NB65)\n"
        "- pos-neg gap >= 0.04\n"
        "\n"
        "If decoded_f1 >= 0.08: K=4 is the sweet spot and unblocks the combined cls + geometry "
        "trade-off. Combine in future runs."
    )
    out = NOTEBOOKS / 'stage2_notebook_68_exp2mmm_anchor_cls_sep_topk4_vfl_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb69():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 69 - Exp2NNN Anchor + cls_sep + 5-epoch head_warmup + full 70K + 14ep\n"
        "\n"
        "**Sequential training: heads first, then backbone.** NB63 (DN-DETR + ultra-stable) showed that "
        "during head_warmup (backbone frozen), val_lane_best_f1 hit 0.630 at epoch 1 -- the highest cls "
        "discrimination ever measured -- then DEGRADED when backbone unfroze at epoch 4. The head's "
        "converged cls patterns were destroyed by backbone updates.\n"
        "\n"
        "Exp2NNN applies this insight to the anchor head: extend head_warmup from 1 epoch (NB62) to "
        "5 epochs. The head fully converges on the frozen-random backbone (still does mask aux training, "
        "still does geometry training within head's params) BEFORE the backbone is allowed to update. "
        "Then 9 epochs of full_finetune with the now-converged head.\n"
        "\n"
        "Single diff vs NB62 (exp57):\n"
        "- `phases.head_warmup until_epoch: 1 -> 5`\n"
        "- `end_epoch: 12 -> 14` (compensate for the longer warmup phase)\n"
        "- `lr_scheduler.warmup_epochs: 1 -> 2`"
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 14 epochs full 70K. ~3-3.5 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp64_rmt_gca_anchor_cls_sep_vfl_long_warmup_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp64_rmt_gca_anchor_cls_sep_vfl_long_warmup_full_data_joint.yaml',
        run_tag='full14', epochs=14, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2NNN\n"
        "\n"
        "Reference NB62 (1-ep head_warmup, 12 ep total): matched_iou=0.550, decoded_f1=0.073, gap=0.045.\n"
        "Reference NB63 (DN-DETR with peak in head_warmup): val_lane_best_f1=0.630 at ep1 (decayed).\n"
        "\n"
        "Pass criteria at epoch 14:\n"
        "- **In epochs 1-5 (head_warmup)**: val_lane_f1 climbing steadily, val_lane_best_f1 should "
        "approach 0.20 by epoch 5 (vs NB62 ep1=0.000).\n"
        "- **In epochs 6-14 (full_finetune)**: cls preserved, geometry climbing as backbone unblocks.\n"
        "- val/matched_line_iou >= 0.55 (preserve NB62 geometry).\n"
        "- val/lane/decoded_f1 >= 0.10 (40 % over NB62; preserved cls + improved geometry).\n"
        "- pos-neg gap >= 0.06."
    )
    out = NOTEBOOKS / 'stage2_notebook_69_exp2nnn_anchor_cls_sep_vfl_long_warmup_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb67, write_nb68, write_nb69):
        out = fn()
        print('wrote:', out)
