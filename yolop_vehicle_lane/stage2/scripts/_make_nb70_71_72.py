"""Generate NB70 (Exp2OOO), NB71 (Exp2PPP), NB72 (Exp2QQQ)."""
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


def _train(cfg, run_tag, epochs, limit_train, limit_val=2000):
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


def write_nb70():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 70 - Exp2OOO High-res mask aux (144x256) + NB62 recipe\n"
        "\n"
        "**Architectural change #1: double the segmentation supervision resolution.** Across NB39-69 "
        "the aux mask was 72x128. CLRKDNet uses 288x800. Doubling our aux_mask_size to 144x256 forces "
        "the model's lane mask decoder + the feature pathway feeding it to encode FINER spatial detail. "
        "This propagates into the per-prior ROI features, which is what the cls head reads from.\n"
        "\n"
        "Hypothesis: finer mask supervision tightens the spatial gradient signal, which improves the "
        "per-prior feature discrimination that has been the cls bottleneck.\n"
        "\n"
        "Single architectural diff vs NB62 (exp57):\n"
        "- `dataset.aux_mask_size: [72, 128] -> [144, 256]`\n"
        "- `model.lane_head.mask_size: [72, 128] -> [144, 256]` (matches)\n"
        "- `loss.lane.w_mask: 1.0 -> 1.5` (leverage the higher-res signal)\n"
        "- This is an ARCHITECTURAL change, not just a hyperparameter knob: the mask decoder upsamples "
        "differently, the dataset loader produces a 144x256 GT mask, and the BCE+Dice loss runs on 4x "
        "more pixels.\n"
        "\n"
        "Reference: HRNet (Sun et al. CVPR 2019) -- maintaining high-resolution representations improves "
        "fine-grained tasks. CLRKDNet itself uses 288x800 mask supervision."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~3-3.5 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp65_rmt_gca_anchor_cls_sep_vfl_hires_mask_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp65_rmt_gca_anchor_cls_sep_vfl_hires_mask_full_data_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2OOO\n"
        "\n"
        "Reference NB62 (72x128 mask aux, w_mask=1.0): matched_iou=0.550, decoded_f1=0.073, "
        "val_lane_f1=0.118, val_lane_best_f1=0.138, val_lane_mask=0.45.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- val/lane/mask (BCE+Dice loss on the high-res mask) should DECREASE from ~0.55 to ~0.40 -- "
        "evidence the model learned fine-grained spatial features.\n"
        "- val/matched_line_iou >= 0.55 (preserve NB62 geometry).\n"
        "- val/lane/decoded_f1 >= 0.08 (10% over NB62).\n"
        "- val/lane_f1 >= 0.13, val/lane_best_f1 >= 0.15.\n"
        "- pos-neg gap >= 0.05.\n"
        "\n"
        "If decoded_f1 jumps >= 0.10: high-res mask aux is the lift; queue for combining with topk_fixed "
        "and longer training."
    )
    out = NOTEBOOKS / 'stage2_notebook_70_exp2ooo_anchor_cls_sep_vfl_hires_mask_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb71():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 71 - Exp2PPP Deeper ROI refinement (4 layers) + 72 sample points\n"
        "\n"
        "**Architectural change #2: deeper per-prior feature aggregation.** The CLRKDLaneHead's ROI "
        "gather samples each prior's curve at `sample_points=36` positions, fuses across scales via "
        "scale_blocks, and runs `roi_refine_layers=3` iterative refinements. Each refinement re-samples "
        "the (refined) curve and updates again. NB62 used 36 sample points and 3 refinement stages.\n"
        "\n"
        "Exp2PPP:\n"
        "- `sample_points: 36 -> 72` -- double the points sampled along each prior's curve. The per-prior "
        "feature vector is now 2x longer before the FC projection, so each prior carries a denser feature "
        "signature.\n"
        "- `roi_refine_layers: 3 -> 4` -- one extra iterative refinement stage. CLRNet's original code "
        "uses 3 stages but reports 4-5 can help on larger datasets. With 4 layers each prior gets one "
        "more chance to re-sample features along its refined curve.\n"
        "\n"
        "Architectural diff vs NB62 (exp57):\n"
        "- `model.lane_head.sample_points: 36 -> 72`\n"
        "- `model.lane_head.roi_refine_layers: 3 -> 4`\n"
        "\n"
        "Both increase the lane head's compute by ~50%. Wall-clock per epoch: ~1.6x NB62's.\n"
        "\n"
        "Reference: CLRNet (Zheng et al. CVPR 2022) -- iterative ROI gather with curve refinement is "
        "the published mechanism. We are extending it slightly."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~3.5-4 hr (deeper head)."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp66_rmt_gca_anchor_cls_sep_vfl_deep_roi_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp66_rmt_gca_anchor_cls_sep_vfl_deep_roi_full_data_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2PPP\n"
        "\n"
        "Reference NB62 (sample_points=36, refine_layers=3): matched_iou=0.550, decoded_f1=0.073, "
        "val_lane_f1=0.118.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- val/matched_line_iou >= 0.58 (deeper refinement gives more iterations of curve update).\n"
        "- val/lane/decoded_f1 >= 0.08 (10 % over NB62).\n"
        "- val/lane_f1 >= 0.13.\n"
        "- pos-neg gap >= 0.05.\n"
        "- val/lane/aux0, aux1, aux2 (intermediate stage losses) should DECREASE monotonically -- "
        "evidence that each refinement stage is producing better curves than the previous.\n"
        "\n"
        "If matched_iou pushes >= 0.60: the geometry was capacity-limited at the ROI feature level, "
        "and the extra refinement stage unlocks it. Combine with high-res mask aux (Exp2OOO) for "
        "compounding gains."
    )
    out = NOTEBOOKS / 'stage2_notebook_71_exp2ppp_anchor_cls_sep_vfl_deep_roi_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb72():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 72 - Exp2QQQ KD from NB62 teacher + K=4 topk_fixed student\n"
        "\n"
        "**Architectural / training change #3: knowledge distillation.** NB62 (decoded_f1=0.073, "
        "matched_iou=0.550) has the best ANCHOR-HEAD geometry on the project. NB68 (K=4 topk_fixed, "
        "decoded_f1=0.067, val_lane_best_f1=0.216) has the best CLS discrimination. They are at the "
        "two ends of the same trade-off.\n"
        "\n"
        "Exp2QQQ: train a student model with NB68's cls recipe (K=4 topk_fixed, cls_sep, VFL), but "
        "AUGMENT the loss with distillation from NB62's geometry-strong teacher. The teacher's cls "
        "and coord_pred are passed through as soft targets via the existing `w_distill > 0` KD path "
        "in FusionLaneLoss. The student should:\n"
        "1. Get NB68's cls discrimination from its own VFL + K=4 supervision\n"
        "2. Get NB62's geometric precision from MSE-aligning its coord_pred to teacher's\n"
        "\n"
        "Diffs vs NB62 (exp57):\n"
        "- `teacher.lane_head_checkpoint: null -> NB62's best.pt`\n"
        "- `loss.lane.w_distill: 0.0 -> 1.0`\n"
        "- `lane_assigner: dynamic_k -> topk_fixed`, `topk_fixed_per_gt: 4` (NB68 cls winner)\n"
        "\n"
        "Reference: Hinton et al. 'Distilling the Knowledge in a Neural Network' (2015); CLRKDNet "
        "uses self-distillation between its own intermediate stages as the namesake mechanism."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "1. Smoke (verify teacher checkpoint loads).\n"
        "2. 12 epochs full 70K. ~3.5-4 hr.\n"
        "\n"
        "NOTE: The teacher checkpoint `exp57_..._best.pt` must exist in Drive. Run NB62 first if not."
    )
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp67_rmt_gca_anchor_cls_sep_vfl_kd_from_nb62_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp67_rmt_gca_anchor_cls_sep_vfl_kd_from_nb62_full_data_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2QQQ\n"
        "\n"
        "Reference NB62 (teacher): matched_iou=0.550, decoded_f1=0.073, val_lane_f1=0.118.\n"
        "Reference NB68 (student's matcher): matched_iou=0.368, val_lane_f1=0.193, decoded_f1=0.067.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- val/lane/distill (the KD loss term) should DECREASE -- evidence the student is mimicking "
        "the teacher.\n"
        "- val/matched_line_iou >= 0.50 (KD pulls geometry up toward teacher's 0.55).\n"
        "- val/lane_f1 >= 0.15 (K=4 matcher gives discrimination, NOT below NB62's 0.118).\n"
        "- val/lane/decoded_f1 >= 0.10 (40% over NB62 if KD genuinely combines the two strengths).\n"
        "- pos-neg gap >= 0.05.\n"
        "\n"
        "If decoded_f1 >= 0.10: KD from the geometry-strong teacher into a cls-strong student does "
        "combine the two trade-off endpoints. This is the architectural/training fusion that pure "
        "config-tuning could not achieve."
    )
    out = NOTEBOOKS / 'stage2_notebook_72_exp2qqq_anchor_cls_sep_vfl_kd_from_nb62_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb70, write_nb71, write_nb72):
        out = fn()
        print('wrote:', out)
