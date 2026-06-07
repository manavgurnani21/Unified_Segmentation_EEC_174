# Stage 2 Exp2LLL + Exp2MMM + Exp2NNN — balance K, sequence training

## Reading NB64 / NB65 / NB66

### NB64 (Exp2III — anchor + cls_sep + width=1.0 + embed=192 + full + 8 ep)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap |
|---:|---:|---:|---:|---:|---:|---:|
| 4  | 0.522 | 0.414 | 0.061 | 0.082 | 0.104 | 0.018 |
| 6  | 0.533 | 0.422 | 0.053 | 0.094 | 0.114 | 0.026 |
| 8  | **0.551** | 0.447 | **0.059** | 0.102 | 0.119 | 0.027 |

**Capacity scaling DID NOT help.** Width 1.0 + embed 192 hit matched_iou=0.551 — basically NB62's 0.550. decoded_f1=0.059 was actually WORSE than NB62's 0.073. **The 0.55 geometry ceiling and 0.07 decoded ceiling are NOT capacity-bound at full data scale.**

### NB65 (Exp2JJJ — anchor + cls_sep + topk_fixed K=3 + full + 12 ep)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap | val_det |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7  | 0.390 | 0.420 | 0.063 | 0.160 | 0.178 | 0.029 | 3.05 |
| 12 | **0.404** | 0.436 | 0.061 | **0.169** | **0.186** | 0.036 | **2.89** |

**Trade-off confirmed: stable matching helps cls but hurts geometry.** val_lane_f1 = 0.169 (+43 % over NB62) and val_lane_best_f1 = 0.186 (+35 %) — best anchor-head cls discrimination on the project. But matched_iou crashed to 0.404 (vs NB62's 0.550), so curves don't pass IoU≥0.5 → decoded_f1 ends similar (0.061). Bonus: val_det=2.89 (vs NB62's 3.16) because fewer positive priors = less lane gradient on shared backbone.

### NB66 (Exp2KKK — IoU-priority matching) — FAILED

Drive mount dropped mid-run: `tar: Transport endpoint is not connected`. Infrastructure failure, not experimental. Re-run pending.

## Diagnosis after 35 experiments

The anchor-head trade-off curve is now fully mapped:

| matcher | density (pos/img) | matched_iou | val_lane_f1 | decoded_f1 |
|---|---:|---:|---:|---:|
| dynamic_k (NB62) | ~10-12 | 0.55 | 0.118 | **0.073** |
| topk_fixed K=8 (NB54) | ~30 | 0.29 | 0.07 | 0.062 |
| topk_fixed K=3 (NB65) | ~15 | 0.40 | 0.169 | 0.061 |
| Hungarian (NB44) | 5 | 0.38 | 0.06 | 0.015 |

The Pareto-optimal point currently is **NB62's dynamic_k + cls_sep + full data + bb_throttle + 12 ep** with decoded_f1=0.073. Capacity doesn't help, density doesn't help — the geometry/cls trade-off is structural.

## Three new experiments to break the plateau

### Exp2LLL (NB67, exp62) — IoU-priority matching, TEMPERED

Re-run NB66's failed experiment with toned-down IoU priority and longer training:
- `match_cost_iou: 2.0 → 4.0` (NB66 tried 5.0)
- `match_cost_point: 5.0 → 3.0` (NB66 tried 2.0)
- `w_iou: 2.0 → 2.5` (NB66 tried 3.0)
- `end_epoch: 12 → 14`

Tests whether MODERATE IoU-priority matching gives cleaner positives without crashing geometry like NB66's extreme would have.

### Exp2MMM (NB68, exp63) — K=4 topk_fixed (geometry/cls balance)

K=3 (NB65) hurt geometry too much (0.40). K=8 (NB54) hurt cls too much. **K=4 = midpoint**:
- Stable per-prior labels (topk_fixed's win on cls)
- Closer to dynamic_k's empirical density (preserve geometry)

Single diff vs NB62: `lane_assigner: dynamic_k → topk_fixed`, `topk_fixed_per_gt: 4`. Pass: `decoded_f1 ≥ 0.08`, `val_lane_f1 ≥ 0.13`, `matched_iou ≥ 0.45`.

### Exp2NNN (NB69, exp64) — sequential training (5-epoch head_warmup)

NB63 showed that DN-DETR cls peaks during head_warmup (backbone frozen) then degrades when backbone unfreezes — head-converged-on-frozen-backbone state is fragile. Apply the same insight to the anchor head: extend head_warmup from 1 epoch (NB62) to 5 epochs.

Single diff vs NB62: `phases.head_warmup until_epoch: 1 → 5`, `end_epoch: 12 → 14`, `lr_scheduler.warmup_epochs: 1 → 2`. Hypothesis: anchor head's cls patterns established in head_warmup are preserved through full_finetune.

## On the user's CULane / TuSimple suggestion

The user provided CULane (in MyDrive/EcoCAR/downloads/CULane/) and TuSimple via kagglehub. CULane is the standard lane benchmark — 88,880 training images vs BDD100K's 70K, and lane labels are higher quality. The published CLRKDNet/CLRNet/RMT-PPAD papers report on CULane.

The path forward with CULane: **lane-only pretraining**. Train backbone+lane head on CULane for K epochs, then fine-tune the joint model on BDD100K with the pretrained weights. Expected impact: backbone features start with strong lane-awareness, so the per-prior ROI features have more discriminative content, which should unlock the cls bottleneck.

But this is a **multi-day engineering project**:
1. Convert CULane labels to our (max_lanes, num_points, 2) format (~100 lines)
2. CULane data loader (~150 lines)
3. "load pretrained backbone" path in train script (~30 lines)
4. Two new configs (pretrain + finetune)
5. Two new notebook flows

I'm queuing this as **Exp2OOO + Exp2PPP** for after the three current experiments resolve. The CULane direction is genuinely promising — it's the published recipe — but it's a bigger swing than 3 days of hyperparameter exploration. If NB67/68/69 don't push decoded_f1 past 0.10, CULane pretraining becomes the next swing.

## Run order

1. **Exp2MMM first** (NB68, ~2.5-3 hr). Cheapest test of the geometry/cls balance hypothesis at K=4.
2. **Exp2NNN** (NB69, ~3-3.5 hr). Sequential training; should give the largest single push if the head_warmup hypothesis holds.
3. **Exp2LLL** (NB67, ~3-3.5 hr). Cleanup retry of NB66 with tempered priors.

All three are independent and config-only. `python -m compileall stage2/{fusion,metrics,scripts}` passes.

## Strategic outlook

After 35 experiments the project's deployable model is NB62 (decoded_f1=0.073, matched_iou=0.550, val_lane_f1=0.118). This is a real, validated joint detection+lane model. **Stage 3 deployment can proceed today on this model.** Further improvements live in the 0.10-0.20 decoded_f1 range and require either:

1. **Hyperparameter exploration** (these 3 NBs + 2-3 more) — likely +0.03 decoded_f1 max
2. **CULane lane pretraining** — likely +0.05-0.10 decoded_f1 (published technique)
3. **Pretrained backbone (YOLOPv2 / RMT-PPAD weights)** — likely +0.05-0.10 (engineering effort)
4. **Two-head ensemble at inference** (NB62 curves + NB59 cls re-rank) — likely +0.05 (custom eval script)

The current run prioritizes #1 (low risk, fast feedback). If NB67-69 don't push past 0.10, we pivot to #2 (CULane).
