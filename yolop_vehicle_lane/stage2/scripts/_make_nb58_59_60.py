"""Generate NB58 (Exp2CCC), NB59 (Exp2DDD), NB60 (Exp2EEE)."""
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
        "import os, sys\n\n"
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


def write_nb58():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 58 - Exp2CCC Anchor + cls_separate_path + topk_fixed K=3 + VFL\n"
        "\n"
        "**The untested config knob.** Across 26 experiments we never enabled `cls_separate_path=True` on "
        "`CLRKDLaneHead`. This flag builds a PARALLEL cls-aggregator pathway: separate `scale_blocks_cls`, "
        "`scale_fusion_cls`, `fc_cls`, `cross_attn_cls`. Cls features flow through their own ROI gather + "
        "cross-attention, disjoint from geometry features.\n"
        "\n"
        "Hypothesis: the anchor head's cls collapse (gap ≤ 0.015 across 11 anchor-head experiments) is "
        "caused by cls and geometry sharing the same per_prior_features. With cls_separate_path=True, "
        "cls has its own parameter budget to learn discriminative features that the geometry losses don't "
        "pull toward 'average lane-y curve' representations.\n"
        "\n"
        "Combined with K=3 topk_fixed (NB55's geometry winner) and VFL.\n"
        "\n"
        "Single new config knob vs NB55 (exp50): `cls_separate_path: false -> true`. Adds ~5% params to "
        "the lane head but no other change."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. `DEBUG_MODE=True` smoke.\n2. `DEBUG_MODE=False` 20 ep limit=3000."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp53_rmt_gca_anchor_cls_sep_topk3_vfl_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp53_rmt_gca_anchor_cls_sep_topk3_vfl_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2CCC\n"
        "\n"
        "Reference NB55 (K=3 topk_fixed, cls_separate_path=False): matched_iou=0.381, gap=0.002, "
        "decoded_f1=0.053, val_lane_best_f1=0.139.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- `pos_score - neg_score >= 0.03` -- the smoking gun. If separating cls features cracks the "
        "anchor head's cls equilibrium, gap should be 10x NB55's.\n"
        "- `val/matched_line_iou >= 0.40` -- preserve NB55's geometry.\n"
        "- `val/lane/decoded_f1 >= 0.07` -- 1.3x NB55.\n"
        "- `val/lane_best_f1 >= 0.20` -- 1.4x NB55.\n"
        "\n"
        "If gap < 0.01: cls_separate_path doesn't help and the anchor head's cls is fundamentally "
        "bottlenecked by the per-prior ROI feature design itself. Confirms we need to abandon the "
        "anchor head for cls and use the query head with DAB+DN.\n"
        "\n"
        "If gap >= 0.05: we have an anchor head that finally does cls. Combine with NB48 full-data "
        "geometry in a future Exp2FFF."
    )
    out = NOTEBOOKS / 'stage2_notebook_58_exp2ccc_anchor_cls_sep_topk3_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb59():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 59 - Exp2DDD DN-DETR + 10ep head_warmup + slow backbone\n"
        "\n"
        "**Save the trapped potential.** NB56 (DN-DETR + 30 ep) hit val_lane_best_f1=0.588 and "
        "pos-neg gap=0.328 at EPOCH 1 of head_warmup (backbone frozen). Both metrics DEGRADED as "
        "training continued past the phase boundary at epoch 4. The peak best_f1 ended up at 0.468 "
        "by epoch 30 -- a 20% regression from the peak.\n"
        "\n"
        "Diagnosis: head_warmup (3 epochs) was too short. The head's cls converged to a strong "
        "discriminative state on the frozen-random backbone, but when backbone unfroze at epoch 4 "
        "the backbone updates destabilized the head's learned cls patterns. With more head_warmup "
        "AND slower backbone unfreeze, the head should retain its peak.\n"
        "\n"
        "Diffs vs NB56 (exp51):\n"
        "- head_warmup until_epoch: 3 -> 10 (head fully converges before backbone is allowed to move)\n"
        "- backbone_lr_mult: 0.1 -> 0.02 (5x slower than NB56, 10x slower than baseline)\n"
        "- end_epoch: 30 -> 20 (we now expect peak earlier with the longer warmup)\n"
        "- warmup_epochs (LR scheduler): 3 (unchanged)\n"
        "\n"
        "Reference for the recipe: DETR's original training (Carion 2020) used `lr_backbone=1e-6` "
        "while main LR was 1e-4 -- 100x slower backbone. We're at 50x slower with lr0=2e-4 + "
        "bb_mult=0.02 -> backbone LR = 4e-6."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke first.\n2. 20 epochs limit=3000."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp54_rmt_gca_query64_dn_vfl_long_warmup_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp54_rmt_gca_query64_dn_vfl_long_warmup_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2DDD\n"
        "\n"
        "Reference NB56 epoch 1 (peak): val_lane_best_f1=0.588, val_lane_f1=0.451, gap=0.328, matched_iou=0.147.\n"
        "Reference NB56 epoch 30: val_lane_best_f1=0.468, gap=0.109, matched_iou=0.139.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **val/lane_best_f1 stays >= 0.50 from epoch 10 onwards** -- the slow backbone preserves "
        "the head's peak cls discrimination.\n"
        "- val/lane_f1 >= 0.40.\n"
        "- pos_score - neg_score >= 0.10 at epoch 20 (NB56 was 0.10).\n"
        "- matched_iou >= 0.20 (DN queries get more epochs to refine geometry).\n"
        "- val_det <= 2.5 (slow backbone should prevent the NB56 epoch-7 det collapse).\n"
        "\n"
        "If val_lane_best_f1 stays high: this is the DEPLOYABLE query-head model. Geometry weak but "
        "cls strong; combine with anchor-head curves at inference in Exp2FFF.\n"
        "\n"
        "If val_lane_best_f1 still degrades: peak-vs-decay isn't about the phase transition; query "
        "embeddings inherently drift. Pivot to EMA model averaging."
    )
    out = NOTEBOOKS / 'stage2_notebook_59_exp2ddd_query64_dn_vfl_long_warmup_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb60():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 60 - Exp2EEE Anchor + VFL + full 70K + bb_throttle + 12 epochs\n"
        "\n"
        "**Push anchor geometry past 0.60.** NB48 (6 ep, full data) hit matched_iou=0.544. "
        "NB57 (6 ep + bb_throttle=0.01) hit 0.553 (project record). Both showed matched_iou "
        "still climbing at the final epoch -- the runs were too short. Exp2EEE extends to 12 "
        "epochs at full data (~105,000 iterations total, 2x NB48/NB57).\n"
        "\n"
        "Same as NB57 but `end_epoch: 6 -> 12`. If matched_iou pushes past 0.60, that's "
        "geometry headroom that puts us in CLRKDNet-CULane competitive territory.\n"
        "\n"
        "If matched_iou plateaus at ~0.56: the geometry is capacity-bound and we need a wider "
        "backbone (NB45 used width 1.0 + 30 ep at limit=3000 and hit 0.525). Combining width=1.0 "
        "with full data + 12 epochs would be the next escalation."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~2.5-3 hr wall-clock."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp55_rmt_gca_anchor_vfl_full_data_long12_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp55_rmt_gca_anchor_vfl_full_data_long12_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2EEE\n"
        "\n"
        "Reference NB57 (6 ep full + bb_throttle): matched_iou=0.553, oracle_f1=0.455, "
        "decoded_f1=0.042, val_det=3.17.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- **val/matched_line_iou >= 0.60** -- decisive geometry improvement; matches CLRKDNet "
        "on CULane.\n"
        "- val/lane/decoded_oracle_f1 >= 0.50.\n"
        "- val/lane/decoded_f1 >= 0.05 (don't regress NB48).\n"
        "- train_lane still decreasing at epoch 12 OR has plateaued (use plateau as signal to stop).\n"
        "\n"
        "If matched_iou >= 0.60: the geometry breakthrough. We have a 'super geometry' model.\n"
        "Combine with NB59's strong-cls model in Exp2FFF (post-hoc cls re-ranking by feeding NB59's "
        "trained query head as a re-ranker over NB60's anchor curves)."
    )
    out = NOTEBOOKS / 'stage2_notebook_60_exp2eee_anchor_vfl_full_data_long12_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb58, write_nb59, write_nb60):
        out = fn()
        print('wrote:', out)
