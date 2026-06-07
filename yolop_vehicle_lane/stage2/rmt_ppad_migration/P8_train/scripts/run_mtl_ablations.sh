#!/usr/bin/env bash
# =============================================================================
# MTL anti-negative-transfer ablation suite (20 epochs, 10k subset).
#
# Tests each of the 5 techniques in isolation against a baseline, to find
# which one (if any) keeps lane curveIoU rising/stable past the ~ep10
# negative-transfer decline instead of peaking then dropping.
#
# PRECONDITIONS (run these first, e.g. via NB92 cells 1-2 on Colab):
#   - 10k subset extracted to /content/bdd_subset_10k (images + labels +
#     lane_targets). The data YAML's `path:` must point there.
#   - Drive mounted (per-epoch best.pt/last.pt + full_train.log mirror to
#     $DRIVE_CKPT/<run_name>/).
#   - Re-synced repo so the trainer/val have the curveIoU + MTL edits.
#
# SUCCESS CRITERION: in each run's metrics table / results.csv, watch
#   metrics/lane_curveIoU(lane). A technique "works" if curveIoU is stable
#   or rising after epoch 10, versus the baseline's peak-then-decline.
#   (lane_f1 may stay 0 the whole time - the model is below the 0.5 bar;
#    curveIoU is the un-saturated signal that actually moves.)
#
# Runs are SEQUENTIAL (one GPU). A failing run logs and the suite continues.
# Total: 6 runs x ~1.5 h ~= 9 h. Comment out rows to run a subset.
# =============================================================================
set -u

# ---- Config (override via env, e.g. REPO_ROOT=... bash run_mtl_ablations.sh) ----
REPO_ROOT="${REPO_ROOT:-/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane}"
MIG="$REPO_ROOT/stage2/rmt_ppad_migration"
TRAIN="$MIG/P8_train/scripts/train_lane_only.py"
MODEL="${MODEL:-$MIG/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml}"
DATA="${DATA:-$MIG/vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml}"
PROJECT="${PROJECT:-/content/runs/mtl_ablation}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-32}"
LR0="${LR0:-4e-4}"
INTERVENE_EP="${INTERVENE_EP:-10}"   # det-decay + freeze trigger epoch

# Shared config = the current NB97 polyline recipe (so the ONLY difference
# between rows is the MTL technique under test).
COMMON=(--mode full --model-yaml "$MODEL" --data-yaml "$DATA"
        --project "$PROJECT" --device "$DEVICE"
        --epochs "$EPOCHS" --batch "$BATCH" --lr0 "$LR0"
        --lane-weights clrkd --diff-clamp 100 --lane-match hungarian
        --fliplr 0.5 --weight-decay 0.05
        --val-period 1 --patience 99 --save-period 20)

cd "$MIG" || { echo "FATAL: cannot cd $MIG"; exit 1; }

run_one () {
  local name="$1"; shift
  echo ""
  echo "==================================================================="
  echo "=== ABLATION: $name   ($(date '+%H:%M:%S'))"
  echo "===   extra flags: $*"
  echo "==================================================================="
  python -u "$TRAIN" "${COMMON[@]}" --name "$name" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[suite] WARNING: '$name' exited rc=$rc - continuing to next ablation."
  fi
}

# 0. Baseline — no MTL technique (this is the curve we must beat: peak~ep10
#    then decline). Everything else is baseline + exactly ONE change.
run_one abl_baseline

# 2. Asymmetric LR — shared backbone+neck at 0.1x the head LR.
run_one abl_asym_lr        --backbone-lr-mult 0.1

# 3. Strengthen GCA — raise the lane gate floor to 0.3 (force task-specific
#    transform so lanes decouple from the detection-dominated trunk).
run_one abl_gca_floor      --seg-gate-floor 0.3

# 4. Dynamic det-loss decay — halve L_det at epoch 10.
run_one abl_det_decay      --det-decay-epoch "$INTERVENE_EP" --det-decay-factor 0.5

# 5. Asymmetric trunk freeze — freeze backbone+neck at epoch 10.
run_one abl_freeze         --freeze-trunk-after "$INTERVENE_EP"

# 1. PCGrad — project conflicting detection gradient off the lane gradient.
#    Last because it's the most expensive (~2x backward). Needs amp off (it is).
run_one abl_pcgrad         --pcgrad

# 6. Auxiliary dense seg — training-only drivable+lane dense supervision to
#    inject dense gradients into the shared trunk. NOTE: the DRIVABLE term
#    only fires if the subset was rebuilt with drivable masks; otherwise this
#    degrades to lane-only dense aux (still a regularizer, weaker). Bypassed
#    entirely at inference.
run_one abl_aux_seg        --use-aux-seg --aux-seg-classes 2 \
                           --aux-drivable-weight 0.5 --aux-lane-weight 0.5

echo ""
echo "==================================================================="
echo "=== SUITE DONE. Compare metrics/lane_curveIoU(lane) across:"
echo "===   $PROJECT/abl_baseline  vs  abl_asym_lr / abl_gca_floor /"
echo "===   abl_det_decay / abl_freeze / abl_pcgrad"
echo "===   Winner = curveIoU stable/rising after epoch $INTERVENE_EP."
echo "==================================================================="
