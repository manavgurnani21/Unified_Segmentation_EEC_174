# Stage 2 Exp2CCC + Exp2DDD + Exp2EEE — separate cls path, save the DN peak, push anchor geometry

## Reading NB55 / NB56 / NB57 — three sharper findings

### NB55 (Exp2ZZ — anchor + topk_fixed K=3 + VFL, 3K)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_best_f1 | gap | val_det | val_map50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.264 | 0.267 | 0.016 | 0.118 | 0.002 | 2.13 | 0.003 |
| 10 | 0.352 | 0.402 | 0.030 | 0.129 | -0.001 | 1.98 | 0.005 |
| 20 | **0.381** | **0.434** | 0.053 | **0.139** | **0.002** | 1.84 | **0.009** |

**Geometry recovered from K=8's collapse** (NB52: matched_iou=0.256). K=3 is the geometry sweet spot for topk_fixed. But cls gap STILL 0.002 — matcher density doesn't unlock cls on the anchor head.

### NB56 (Exp2AAA — DN-DETR K=64 + 30 epochs + det rescue, 3K)

| ep | matched_iou | val_lane_f1 | val_lane_best_f1 | pos / neg | **gap** | val_det |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.147 | 0.451 | **0.588** ← peak | 0.554 / 0.226 | **0.328** ← project record | 2.11 |
| 6  | 0.112 | 0.128 | 0.170 | 0.288 / 0.289 | -0.001 | 3.31 |
| 12 | 0.120 | 0.553 | 0.553 | 0.373 / 0.230 | 0.143 | 3.36 |
| 30 | 0.139 | 0.466 | 0.468 | 0.360 / 0.251 | 0.109 | 3.23 |

**Two huge findings here**:

1. **val_lane_best_f1 = 0.588 at epoch 1.** This is the HIGHEST cls discrimination ever recorded on the project, by a wide margin. The DN-DETR architecture CAN do 60% F1 on lanes.

2. **The peak was at HEAD_WARMUP phase (backbone frozen)**. Once the backbone unfroze at epoch 4, val_det immediately crashed (2.13→3.31) and the head's cls discrimination dropped (gap 0.328 → -0.001 by epoch 6, recovered partially later but never returned to peak). The peak is a "trapped potential" that the standard training schedule destroys.

### NB57 (Exp2BBB — anchor + VFL + full 70K + backbone_lr_mult=0.01)

| ep | matched_iou | oracle_f1 | decoded_f1 | gap | val_det | val_map50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.437 | 0.303 | 0.002 | -0.003 | 3.36 | 0 |
| 3  | 0.535 | 0.440 | 0.050 | 0.013 | 3.26 | 0.0009 |
| **6** | **0.553** ✓ | 0.455 | 0.042 | 0.015 | **3.17** | **0** |

**matched_iou=0.553 — new project geometry record** (vs NB48's 0.544 and NB54's 0.549). But:
- backbone_lr_mult=0.01 did NOT fix det at full data (val_det stuck at 3.17, val_map50 still 0).
- Cls gap unchanged at 0.015.

**The "joint conflict at full data" hypothesis is REJECTED**. backbone throttling 10× didn't help det. The full-data det issue is something else (likely evaluation-side: LIMIT_VAL=2000 has harder samples; mAP threshold at 0.5 IoU is too strict for our compressed-RMT detector at full data; or the mAP metric saturates to 0 with sparse predictions on the larger val set).

## Cumulative diagnosis after 26 experiments

| problem | architecture that solves it | current best | who CANNOT solve it |
|---|---|---|---|
| **Geometry** | anchor 192 + dynamic-k | 0.553 (NB57) | query heads (matched_iou stuck at 0.15-0.27) |
| **Cls discrimination** | query 64 + DAB + DN + VFL | val_lane_best_f1=0.588 (NB56 ep1) | anchor heads (gap ≤ 0.015 across 11 experiments) |
| **Joint conflict @ full data** | **unresolved** | val_det floor ≈ 3.10, val_map50 ≈ 0 | head re-weighting (NB48/51/54), backbone throttling (NB57) |

**The cls collapse on the anchor head is intrinsic to the per-prior ROI feature design.** 11 experiments, 4 loss formulations, 3 matcher schemes, 2 capacity scales — all land at gap ≤ 0.015. The per-prior features simply do not encode "this prior matches a real lane" vs "this prior is somewhere lane-y." Geometry losses pull the features toward smooth lane-y representations; cls can't differentiate.

## Three new experiments

### Exp2CCC (NB58, exp53) — the untested config knob

`cls_separate_path=True` on `CLRKDLaneHead`. Builds a parallel cls-aggregator pathway (`scale_blocks_cls`, `scale_fusion_cls`, `fc_cls`, `cross_attn_cls`) so cls features flow through their own ROI gather + cross-attention, disjoint from geometry features. Combined with K=3 topk_fixed (NB55's geometry winner) and VFL.

**This is the single config knob we have not tested in 26 experiments.** If cls's per-prior representation is the bottleneck, giving it disjoint feature pathway parameters might break the equilibrium. Single new field vs NB55: `cls_separate_path: false → true`.

### Exp2DDD (NB59, exp54) — save the DN trapped potential

Recipe from NB56's diagnosis:
- `head_warmup` until epoch 10 (vs NB56's 3) — head fully converges before backbone moves
- `backbone_lr_mult: 0.1 → 0.02` (5× slower than NB56, 50× slower than baseline)
- `end_epoch: 30 → 20`

This is the standard DETR recipe (`lr_backbone=1e-6` while main LR is 1e-4; we're at 4e-6 backbone with 2e-4 main → exactly the DETR ratio of 50×). Hypothesis: the head's epoch-1 cls peak persists when the backbone updates carefully instead of catastrophically.

### Exp2EEE (NB60, exp55) — push anchor geometry past 0.60

Same as NB57 but `end_epoch: 6 → 12`. NB48 and NB57 both showed matched_iou still climbing at epoch 6 — runs were too short. Doubling to 12 epochs (~105k iters) tests if the geometry crosses 0.60.

If matched_iou ≥ 0.60: we have CLRKDNet-CULane geometry. If it plateaus at ~0.56: capacity-bound (width=0.5 ceiling).

## Code state

No code changes needed — `cls_separate_path`, `LaneQueryHeadAnchorDN`, `backbone_lr_mult` all already exist. 3 new configs, 3 new notebooks, NB08 patched. `python -m compileall stage2/{fusion,metrics,scripts}` passes.

## Run order — independent

1. **Exp2CCC** (NB58, ~30 min, 3K). Tests the only untouched anchor-head knob. If it works, the anchor head finally does cls.
2. **Exp2DDD** (NB59, ~30 min, 3K). Tests if the DN peak can be preserved with slow backbone.
3. **Exp2EEE** (NB60, ~2.5-3 hr, full 70K). Tests geometry ceiling.

## Pass criteria

### Exp2CCC at epoch 20
- **`pos_score - neg_score ≥ 0.03`** (10× NB55's 0.002 — the smoking gun for cls_separate_path)
- `val/matched_line_iou ≥ 0.40` (preserve NB55's geometry)
- `val/lane/decoded_f1 ≥ 0.07`

### Exp2DDD at epoch 20
- `val/lane_best_f1 ≥ 0.50` STAYS that high from epoch 10 onward (vs NB56 dropping to 0.47)
- `pos_score - neg_score ≥ 0.10` at epoch 20
- `matched_iou ≥ 0.20`

### Exp2EEE at epoch 12 — geometry ceiling
- **`val/matched_line_iou ≥ 0.60`** (CLRKDNet-CULane territory)
- `val/lane/decoded_oracle_f1 ≥ 0.50`

## Strategic context

After 26 experiments my diagnosis is sharper: **anchor-head cls is intractable; query-head geometry is intractable**. They have orthogonal strengths and the previous fusion attempt (NB49 HybridPriorQueryHead) failed because stage 2 regenerated geometry from scratch. The endgame architecture is a **late-fusion ensemble**:

1. Train an anchor model with strong geometry (NB57/NB60 recipe) → champion curves.
2. Train a DN-DETR model with strong cls (NB59 recipe if it works) → champion scores.
3. At inference: take anchor's top-K curves, rank by query head's cls applied via cross-attention onto the curves.

This is essentially "two heads, one inference," and it bypasses the architectural bottleneck of trying to make one head do both. If Exp2CCC unlocks anchor cls, we won't need this ensemble. If not, Exp2DDD's preserved-peak model + Exp2EEE's strong-geometry model become the two components.

## Why I'm not pursuing GELAN/PGI/AdaBins yet

These remain queued. My empirical data says the lane head is the bottleneck, not the backbone or auxiliary heads. Once Exp2CCC/DDD/EEE settle the cls-vs-geometry question, GELAN-PGI would be the right next move for stabilizing the backbone for the depth head.
