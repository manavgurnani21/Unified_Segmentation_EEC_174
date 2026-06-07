# MTL negative-transfer ablation suite — findings + plan

Date 2026-05-30. From inspecting the four synced runs
(`polyline_baseline_small`, `clr_lane_polyline_full_v2`,
`bezier_cubic_no_lcm`, `bezier_lcm_gamma001`) and building the 5-technique
anti-negative-transfer toolkit.

## 1. What the synced runs actually showed

**Bezier t-range fix VERIFIED** (`bezier_cubic_no_lcm`, full 30-ep re-run):
`lane_bezier_span` = **0.975 from epoch 1** (was the 0.05 stub), and lane
IoU **~doubled** (old peak 0.024 → 0.049). The init fix worked. (LCM row
re-confirms LCM hurts: its span collapses 0.98→0.73, IoU drops to 0.0035.)

**The "severe decline" is much milder than the premise — in the
regularized runs it's a PLATEAU, not a steep drop:**

| run | lane-IoU trajectory | shape |
|---|---|---|
| `polyline_baseline_small` (30ep,10k) | 0.013 → 0.072 @ep25 → 0.071 @ep30 | plateau |
| `bezier_cubic_no_lcm` (30ep) | monotonic rise to 0.049 | no decline |
| `clr_lane_polyline_full_v2` (5ep) | 0.023→0.041→0.067→0.076→0.075 | still rising |

The steep "peak ep5 then decline" was real in the EARLIER un-regularized
NB97 (clamp=none). clamp=100 + fliplr=0.5 + wd=0.05 + early-stop already
softened it to a plateau. Detection mAP50 actually erodes more than lanes
(0.702→0.673 in the 30-ep run).

**The real wall is precision, not the decline: `lane_f1 = 0.0000` in every
epoch of every run.** Geometry stats (startx≈0.37, theta≈0.49, length≈0.09)
freeze after ~ep5. Lanes get roughly placed and stop sharpening — none
clears the per-lane IoU 0.5 bar. The 5 MTL techniques target the decline;
**none directly fixes this precision wall** (that needs the curve-sampling
loss / higher-res rasterization / more capacity — tracked separately).

**Blind spot:** `curveIoU` was empty in these runs (they predate the
metric). Re-sync is required so the ablations report the un-saturated
signal we judge success by.

## 2. The toolkit (all in `training_utils/tools/mtl_techniques.py`, default OFF)

| # | Technique | Flag | Where it hooks |
|---|---|---|---|
| 1 | PCGrad (grad surgery) | `--pcgrad` | trainer loop reads env `MTL_PCGRAD`; reuses `get_backbone_params` + per-task `self.loss=[L_det,da_seg,ll_seg]`; projects det(0) off lane(2) when conflicting. ~2× backward. **Needs amp=False** (guarded). |
| 2 | Asymmetric LR | `--backbone-lr-mult 0.1` | monkeypatches `MTDETRTrainer.build_optimizer` to split backbone (layers 0..27) into its own group at lr×mult, before the scheduler snapshots base_lrs. |
| 3 | Strengthen GCA | `--seg-gate-floor 0.3` | raises `LiteDynamicGate.clamp_min` on the lane path (was 0.05) so lanes inject more task-specific transform. |
| 4 | Dynamic det decay | `--det-decay-epoch 10 --det-decay-factor 0.5` | callback flips env `DET_LOSS_SCALE`; `tasks.py` multiplies `L_det` by it. |
| 5 | Asymmetric freeze | `--freeze-trunk-after 10` | (already existed) freezes backbone+neck `.eval()`+`requires_grad_(False)` at epoch N. |

Re-scoping notes: #3 was requested as "add an adapter/gate" but that block
**already exists** (the GCA `TaskAdapterLite`+`LiteDynamicGate`) — so we
strengthen it rather than duplicate. #5 was already implemented.

Validation done locally (no GPU): all files compile; PCGrad projection math
verified via numpy mirror (projection removes the lane-conflicting
component, leaves aligned/orthogonal grads untouched).

## 3. How to run

`P8_train/scripts/run_mtl_ablations.sh` — 20 epochs, 10k subset, batch 32,
interventions at epoch 10. Baseline + one technique per run (6 runs, ~9 h).
Precondition: 10k subset at `/content/bdd_subset_10k` (NB92 cells 1-2) and a
Drive re-sync (for curveIoU + the MTL edits).

## 4. Success criterion

Per run, watch `metrics/lane_curveIoU(lane)` (the un-saturated column;
`lane_f1` will likely stay 0 — below the 0.5 bar). A technique **works** if
curveIoU is **stable or rising after epoch 10** vs the baseline's
peak-then-erode. Rank, take the winner (or stack the top 2) to full data.

## 5. Honest expectations
- These fight the (now-mild) decline. If the baseline only plateaus, the
  techniques may show small deltas — that's still a useful result.
- `lane_f1=0` (precision) is the bigger problem and is **out of scope** for
  this suite; don't expect F1 to lift off 0 from MTL changes alone.
- `--det-decay` can hurt detection mAP (prior exp01-71). Watch mAP50 too.
