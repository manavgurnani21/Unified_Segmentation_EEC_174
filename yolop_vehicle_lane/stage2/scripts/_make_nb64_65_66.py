"""Generate NB64 (Exp2III), NB65 (Exp2JJJ), NB66 (Exp2KKK)."""
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


def write_nb64():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 64 - Exp2III Anchor + cls_sep + width=1.0 + full 70K + 8ep\n"
        "\n"
        "**Push capacity from NB62's record.** NB62 (width=0.5, full 70K, cls_sep, 12 ep) hit "
        "decoded_f1=0.073, matched_iou=0.550, gap=0.045 -- NEW PROJECT RECORDS. The geometry "
        "plateaued at 0.55 with width=0.5. NB45 (width=1.0 at limit=3000) hit matched_iou=0.525, "
        "showing the capacity headroom.\n"
        "\n"
        "Exp2III: combine NB62's recipe with width=1.0 + embed_dim=192. Hypothesis: 2x backbone "
        "capacity + 1.5x lane head capacity unblocks both geometry (matched_iou past 0.60) and "
        "cls (gap past 0.06).\n"
        "\n"
        "Diffs vs NB62 (exp57):\n"
        "- `model.width: 0.5 -> 1.0`\n"
        "- `lane_head.embed_dim: 128 -> 192`\n"
        "- `lane_head.roi_mid_channels: 48 -> 64`\n"
        "- `end_epoch: 12 -> 8` (1.5x slower per epoch -> 8 epochs ~ NB62's 12-epoch wall clock)\n"
        "\n"
        "GPU mem: NB60 used 10.4 GB. width=1.0 should land around 25-30 GB on the 95 GB Pro 6000."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke first (new wider model).\n2. 8 epochs full data. ~3-3.5 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp59_rmt_gca_anchor_cls_sep_vfl_w1_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp59_rmt_gca_anchor_cls_sep_vfl_w1_full_data_joint.yaml',
        run_tag='full8', epochs=8, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2III\n"
        "\n"
        "Reference NB62 (width=0.5, cls_sep, full, 12 ep): matched_iou=0.550, decoded_f1=0.073, "
        "gap=0.045, val_lane_f1=0.118.\n"
        "\n"
        "Pass criteria at epoch 8:\n"
        "- **val/matched_line_iou >= 0.60** -- width=1.0 unblocks geometry past NB62's 0.55 plateau.\n"
        "- val/lane/decoded_f1 >= 0.10 (40% over NB62).\n"
        "- pos-neg gap >= 0.06.\n"
        "- val/lane_f1 >= 0.15."
    )
    out = NOTEBOOKS / 'stage2_notebook_64_exp2iii_anchor_cls_sep_vfl_w1_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb65():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 65 - Exp2JJJ Anchor + cls_sep + topk_fixed K=3 + full 70K + 12ep\n"
        "\n"
        "**Stable matching on top of NB62's recipe.** NB55 showed K=3 topk_fixed gives "
        "matched_iou=0.381 at limit=3000 (recovered from K=8's collapse). The stable labels "
        "(no batch flicker) should help cls separation when combined with the cls_sep "
        "feature pathway and full data.\n"
        "\n"
        "Single diff vs NB62 (exp57): `lane_assigner: dynamic_k -> topk_fixed`, "
        "`topk_fixed_per_gt: 3`. All else identical (width=0.5, cls_sep=true, full 70K, "
        "12 epochs, bb_throttle=0.01).\n"
        "\n"
        "Tests whether stable per-prior labels combine constructively with cls_sep to give "
        "more cls discrimination than either alone."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~2.5-3 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp60_rmt_gca_anchor_cls_sep_topk3_vfl_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp60_rmt_gca_anchor_cls_sep_topk3_vfl_full_data_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2JJJ\n"
        "\n"
        "Reference NB62 (dynamic_k + cls_sep + full 12ep): matched_iou=0.550, decoded_f1=0.073, "
        "gap=0.045, val_lane_f1=0.118.\n"
        "Reference NB55 (topk_fixed K=3, 3K, NO cls_sep): matched_iou=0.381, gap=0.002.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- val/matched_line_iou >= 0.45 (some geometry loss vs NB62 expected from K=3 dilution).\n"
        "- **pos-neg gap >= 0.08** (2x NB62; topk_fixed's stable labels + cls_sep's disjoint "
        "parameters should multiply).\n"
        "- val/lane/decoded_f1 >= 0.08.\n"
        "- val/lane_f1 >= 0.20."
    )
    out = NOTEBOOKS / 'stage2_notebook_65_exp2jjj_anchor_cls_sep_topk3_vfl_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb66():
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 66 - Exp2KKK Anchor + cls_sep + IoU-priority matching + full 70K + 12ep\n"
        "\n"
        "**Cleaner matching for the cls_sep recipe.** NB62's matching cost weights were "
        "`match_cost_point=5.0, match_cost_iou=2.0` -- point distance heavily favored over "
        "IoU. This may match priors that are coordinate-close but have LOW IoU with the GT, "
        "producing noisy cls supervision.\n"
        "\n"
        "Exp2KKK shifts cost weighting to favor IoU: `match_cost_point: 5.0 -> 2.0`, "
        "`match_cost_iou: 2.0 -> 5.0`. Also bumps `w_iou: 2.0 -> 3.0` to reinforce that the "
        "matched priors actually have high LineIoU with their assigned GT.\n"
        "\n"
        "Hypothesis: cleaner positive matches (higher IoU on matched priors) gives cls a more "
        "discriminative signal. Combined with cls_sep's disjoint feature pathway, gap should "
        "exceed NB62's 0.045."
    )
    nb['cells'][1]['source'] = "### Run mode\n1. Smoke.\n2. 12 epochs full 70K. ~2.5-3 hr."
    nb['cells'][3]['source'] = _smoke('stage2/configs/exp61_rmt_gca_anchor_cls_sep_vfl_iou_match_full_data_joint.yaml')
    nb['cells'][4]['source'] = _train(
        'stage2/configs/exp61_rmt_gca_anchor_cls_sep_vfl_iou_match_full_data_joint.yaml',
        run_tag='full12', epochs=12, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2KKK\n"
        "\n"
        "Reference NB62 (cost_point=5, cost_iou=2): matched_iou=0.550, decoded_f1=0.073, "
        "gap=0.045.\n"
        "\n"
        "Pass criteria at epoch 12:\n"
        "- val/matched_line_iou >= 0.55 (preserved or improved by IoU-priority matching).\n"
        "- val/lane/decoded_f1 >= 0.08 (beat NB62).\n"
        "- **pos-neg gap >= 0.06** (cleaner positives give cls more signal).\n"
        "- val/lane_f1 >= 0.15."
    )
    out = NOTEBOOKS / 'stage2_notebook_66_exp2kkk_anchor_cls_sep_vfl_iou_match_full_data_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb64, write_nb65, write_nb66):
        out = fn()
        print('wrote:', out)
