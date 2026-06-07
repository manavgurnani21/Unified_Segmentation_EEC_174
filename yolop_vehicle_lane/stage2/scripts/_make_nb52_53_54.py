"""Generate NB52 (Exp2WW), NB53 (Exp2XX), NB54 (Exp2YY).

Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_make_nb52_53_54.py
"""
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


_SMOKE = (
    "from pathlib import Path\n"
    "import os, sys\n"
    "\n"
    "CONFIG = '{config}'\n"
    "LOG_FILE = os.path.join(LOG_DIR, f'{{Path(CONFIG).stem}}_smoke.log')\n"
    "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
)


def _train(config: str, run_tag: str, epochs: int, limit_train, limit_val: int = 1000) -> str:
    if limit_train is None:
        limit_block = "LIMIT_TRAIN = None\n"
    else:
        limit_block = f"LIMIT_TRAIN = {limit_train}\n"
    return (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        f"CONFIG = '{config}'\n"
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


def write_nb52() -> Path:
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 52 - Exp2WW Anchor + topk-fixed K=8 matching + VFL\n"
        "\n"
        "**The matcher fix the cls collapse really needed.** Across NB39-46 (focal/ASL/QFL/VFL on the anchor "
        "head) the cls stayed at uniform sigmoid (gap <= 0.015) regardless of loss formulation. NB44 (Hungarian "
        "1-to-1 on anchor) crashed geometry to matched_iou=0.38. NB47 (Hungarian on K=64 query head) FINALLY "
        "broke the cls collapse (gap=0.099) -- but had weak geometry (matched_iou=0.27).\n"
        "\n"
        "Diagnosis: dynamic-k matching's K is **estimated** from sum(top-K IoUs), so it varies batch to batch. "
        "Same prior gets K=2 in one batch, K=4 in another, sometimes 0. Per-prior labels FLICKER and cls "
        "converges to uniform sigmoid as the only stable equilibrium under noisy labels.\n"
        "\n"
        "Hungarian fixes the flickering but with K=1 per GT, only ~5 priors per image get positive geometry "
        "supervision -- starvation that crashed NB44's matched_iou.\n"
        "\n"
        "**Exp2WW: fix-K-by-cost matching.** Each GT lane takes the top K=8 priors by cost, with conflict "
        "resolution (each prior matches at most one GT). Properties:\n"
        "- K=8 priors per GT * 5 GT = ~40 positives per image (anchor-style dense geometry supervision).\n"
        "- Each prior's label is deterministic given the cost matrix -- no IoU-sum random rounding.\n"
        "- Combined with VFL on continuous LineIoU regression target.\n"
        "\n"
        "Code: new `_topk_fixed_match` in losses.py + `lane_assigner='topk_fixed'` + `topk_fixed_per_gt=8` "
        "config. Backwards compatible with all prior runs."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "1. `DEBUG_MODE = True` smoke first (new matcher path).\n"
        "2. `DEBUG_MODE = False` for 20-epoch short run at limit=3000.\n"
        "3. ~30 min wall-clock with AMP."
    )
    nb['cells'][3]['source'] = _SMOKE.format(config='stage2/configs/exp47_rmt_gca_anchor_topk_fixed_vfl_joint.yaml')
    nb['cells'][4]['source'] = _train(
        config='stage2/configs/exp47_rmt_gca_anchor_topk_fixed_vfl_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2WW training\n"
        "\n"
        "Reference NB48 (anchor + dynamic-k + VFL + 70K data): matched_iou=0.544, decoded_f1=0.05, gap=0.015.\n"
        "Reference NB47 (K=64 query + Hungarian + VFL): val_lane_f1=0.246, gap=0.099, matched_iou=0.27.\n"
        "\n"
        "Pass criteria at epoch 20 (limit=3000):\n"
        "- **`pos_score - neg_score >= 0.05`** -- stable labels should give cls room to discriminate.\n"
        "- **`val/matched_line_iou >= 0.45`** -- preserve anchor-head geometry (denser supervision than NB44 "
        "Hungarian's 0.38).\n"
        "- **`val/lane/decoded_f1 >= 0.10`** -- 2x NB48; cls finally ranks correctly.\n"
        "- **`val/lane_f1 >= 0.10` and `val/lane_best_f1 >= 0.20`** -- legacy F1 metrics show real "
        "discrimination (vs anchor head's 0.0 across NB39-46).\n"
        "\n"
        "Failure signals:\n"
        "- gap < 0.02: stable labels alone aren't enough; per-prior features fundamentally non-discriminative. "
        "Move to query-based heads (Exp2XX).\n"
        "- matched_iou < 0.35: K=8 conflict resolution starves geometry; lower to K=6.\n"
        "- gap >= 0.05 AND decoded_f1 >= 0.10: matcher was the bug. Combine with NB48's full data in Exp2YY-style follow-up."
    )
    out = NOTEBOOKS / 'stage2_notebook_52_exp2ww_anchor_topk_fixed_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb53() -> Path:
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 53 - Exp2XX K=64 query head + DAB anchor + DN denoising + VFL\n"
        "\n"
        "**Stabilized version of NB47.** NB47 (K=64 query + Hungarian + VFL) was the cls breakthrough but its "
        "gap PEAKED at epoch 9 (0.099) then degraded to 0.063 by epoch 20 -- classic DETR-style query "
        "oscillation. NB50 (same head + full 70K data) collapsed completely (matched_iou=0.013) -- the queries "
        "couldn't ground themselves in the wider data distribution.\n"
        "\n"
        "Both failure modes are addressed by DAB-DETR + DN-DETR techniques:\n"
        "\n"
        "- **DAB anchors**: each query has a learnable (start_y, start_x, theta) anchor parameter that biases "
        "its spatial attention. Queries can't all converge to the same point. Stabilizes the geometry.\n"
        "- **Denoising queries** (DN-DETR): during training, also pass `dn_num_groups=4` noised copies of every "
        "GT lane through the decoder. Each noised query must reconstruct its noiseless GT, which provides "
        "DENSE gradient signal that prevents query collapse. Standard cls/regr losses on the matched queries "
        "still apply.\n"
        "\n"
        "`LaneQueryHeadAnchorDN` already implements both. Combined with VFL on matched_existence + Hungarian "
        "1-to-1, this is the modern DETR recipe applied to lanes.\n"
        "\n"
        "Reference: Liu et al. 'DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR' (ICLR 2022); "
        "Li et al. 'DN-DETR: Accelerate DETR Training by Introducing Query DeNoising' (CVPR 2022)."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "1. `DEBUG_MODE = True` smoke first.\n"
        "2. `DEBUG_MODE = False` for 20 epochs limit=3000.\n"
        "3. ~30-35 min wall-clock with AMP (DN queries add ~10% per step)."
    )
    nb['cells'][3]['source'] = _SMOKE.format(config='stage2/configs/exp48_rmt_gca_query_anchor_dn_vfl_joint.yaml')
    nb['cells'][4]['source'] = _train(
        config='stage2/configs/exp48_rmt_gca_query_anchor_dn_vfl_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2XX training\n"
        "\n"
        "Reference NB47 (K=64 query plain): peak gap 0.099 at ep9, decay to 0.063 at ep20. matched_iou=0.27.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **gap doesn't decay**: pos-neg gap STABLE >= 0.08 from epoch 10 onwards (vs NB47's degradation).\n"
        "- **matched_iou >= 0.35** -- DAB anchors give queries spatial bias they previously had to discover.\n"
        "- **val/lane_f1 >= 0.30** and **val/lane_best_f1 >= 0.40** -- both above NB47.\n"
        "- **val/lane/decoded_f1 >= 0.10** (5x NB47).\n"
        "\n"
        "Failure signals:\n"
        "- gap < 0.05: DAB anchors didn't help; query head fundamentally limited at K=64.\n"
        "- matched_iou < 0.20: DN queries are dragging the geometry. Lower `dn_num_groups` to 2.\n"
        "- decoded_f1 ~ NB47: stable plateau but no improvement. Pivot to anchor-head topk_fixed (Exp2WW)."
    )
    out = NOTEBOOKS / 'stage2_notebook_53_exp2xx_query64_dn_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb54() -> Path:
    nb = _new_nb(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 54 - Exp2YY Anchor + topk-fixed + VFL + full 70K + det rescue\n"
        "\n"
        "**Combine all the wins on the full dataset.** This config aggregates everything we've learned:\n"
        "- Anchor head with 192 priors (NB48 geometry champion: matched_iou=0.544).\n"
        "- topk-fixed K=8 matching (Exp2WW's stability fix; replaces dynamic-k).\n"
        "- VFL on continuous LineIoU regression target (Exp2QQ recipe).\n"
        "- Full 70K BDD100K split (NB48 / NB51 scale).\n"
        "- Aggressive det rescue: `lambda_det=3.0`, `lambda_lane=0.5`, `use_uncertainty=False` (NB51 used 2.0/0.7 "
        "and got minor improvement; this pushes harder).\n"
        "- 6 epochs (full-data convergence is ~5 epochs based on NB48).\n"
        "\n"
        "If Exp2WW (limit=3000) shows the topk-fixed matcher fixes the cls collapse on the anchor head, AND "
        "Exp2YY (this notebook, full data) inherits that win plus NB48-level geometry plus rescued det, this "
        "is the final stable model for Stage 3 deployment."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "1. `DEBUG_MODE = True` smoke.\n"
        "2. `DEBUG_MODE = False` for 6-epoch full-dataset run (~60-80 min).\n"
        "3. Independent of all prior NBs. Run after Exp2WW (NB52) confirms the matcher fix works.\n"
        "4. If OOM at full data, drop batch_size to 6."
    )
    nb['cells'][3]['source'] = _SMOKE.format(
        config='stage2/configs/exp49_rmt_gca_anchor_topk_fixed_vfl_full_data_det_rescue_joint.yaml'
    )
    nb['cells'][4]['source'] = _train(
        config='stage2/configs/exp49_rmt_gca_anchor_topk_fixed_vfl_full_data_det_rescue_joint.yaml',
        run_tag='full6', epochs=6, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2YY training\n"
        "\n"
        "Reference NB48 (anchor + dynamic-k + VFL + 70K data): matched_iou=0.544, decoded_f1=0.05, gap=0.015, "
        "**val_det=3.16, val_map50=0**.\n"
        "Reference NB51 (same + lambda_det=2.0): no improvement on det.\n"
        "\n"
        "Pass criteria at epoch 6:\n"
        "- **`val_det <= 2.5` and `val/det/map50 >= 0.005`** -- harder det rescue (lambda_det=3.0) actually trains det.\n"
        "- **`val/matched_line_iou >= 0.50`** -- preserve geometry (acceptable small regression from NB48's 0.544).\n"
        "- **`pos_score - neg_score >= 0.05`** -- topk-fixed matcher gives cls room to discriminate even at full data.\n"
        "- **`val/lane/decoded_f1 >= 0.10`** -- 2x NB48; cls finally ranks correctly at full data scale.\n"
        "- `train/grad_cosine_epoch_mean >= 0` across most epochs.\n"
        "\n"
        "Failure signals:\n"
        "- val_det still >= 3.0: lambda_det=3.0 not enough; the joint conflict at full data scale is structural. "
        "Pivot to PCGrad gradient surgery in Exp2ZZ.\n"
        "- gap < 0.02 even with topk-fixed: matcher fix doesn't translate to full-data scale; the per-prior "
        "feature representation is the structural bottleneck. Pivot to query head (Exp2XX) at full data.\n"
        "- matched_iou drops below 0.45: lambda_lane=0.5 too aggressive; raise to 0.7."
    )
    out = NOTEBOOKS / 'stage2_notebook_54_exp2yy_anchor_topk_fixed_vfl_full_data_det_rescue_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb52, write_nb53, write_nb54):
        out = fn()
        print('wrote:', out)
