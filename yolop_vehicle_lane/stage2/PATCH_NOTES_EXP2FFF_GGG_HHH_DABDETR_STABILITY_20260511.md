# Stage 2 Exp2FFF + Exp2GGG + Exp2HHH — isolate joint conflict, scale anchor-cls knob, stabilize DAB-DETR

## What NB58 / NB59 / NB60 actually showed

### NB58 (Exp2CCC — anchor + cls_separate_path=True + K=3 topk_fixed + VFL, 3K)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.263 | 0.266 | 0.020 | 0.000 | 0.147 | 0.004 |
| 10 | 0.340 | 0.380 | 0.029 | 0.013 | 0.137 | 0.004 |
| **20** | 0.374 | 0.444 | 0.047 | **0.099** | 0.147 | **0.011** |

**Marginal but real improvement.** vs NB55 (same config without cls_separate_path): val_lane_f1 = 0.099 (vs 0.000), gap = 0.011 (vs 0.002 — 5.5× improvement). The disjoint cls feature pathway helps the anchor head's cls, but the absolute level is still poor (val_lane_best_f1 = 0.147 vs query head's 0.62+). The anchor's per-prior ROI representation is intrinsically bandwidth-limited.

### NB59 (Exp2DDD — DN-DETR + 10ep head_warmup + bb_lr_mult=0.02, 3K)

| ep | matched_iou | val_lane_f1 | val_lane_best_f1 | pos / neg | gap | val_det |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.154 | 0.458 | 0.573 | 0.529 / 0.220 | 0.309 | 2.12 |
| 2  | 0.078 | 0.272 | 0.317 | 0.343 / 0.224 | 0.119 | 2.16 |
| **3** | 0.091 | **0.623** ✓ | **0.646** ✓ | 0.548 / 0.226 | **0.322** ✓ | 2.20 |
| 4  | 0.092 | 0.277 | 0.353 | 0.345 / 0.222 | 0.123 | 2.19 |
| 11 | 0.140 | 0.472 | **0.594** | 0.514 / 0.253 | 0.261 | 3.36 |
| 13 | 0.153 | 0.502 | 0.551 | 0.453 / 0.256 | 0.197 | 3.36 |
| 15 | 0.187 | 0.222 | 0.256 | 0.322 / 0.272 | 0.050 | 3.36 |
| 20 | 0.221 | 0.240 | 0.297 | 0.342 / 0.288 | 0.054 | 3.37 |

**NEW PROJECT RECORD at epoch 3: val_lane_f1=0.623, val_lane_best_f1=0.646, gap=0.322.** Beats NB56's previous record (0.466 / 0.588). But the cls **OSCILLATES wildly**: across 20 epochs val_lane_f1 ranges 0.22-0.62. Backbone unfroze at epoch 11 (full_finetune), val_det immediately crashed from 2.20 → 3.36. matched_iou stayed weak (0.22 at epoch 20).

### NB60 (Exp2EEE — anchor + VFL + full 70K + bb_throttle + 12 epochs)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_best_f1 | gap | val_det | val_map50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5  | 0.542 | 0.458 | 0.053 | 0.107 | 0.018 | 3.35 | 0.0003 |
| 8  | 0.546 | 0.469 | 0.049 | 0.110 | 0.018 | 3.35 | 0.0003 |
| 11 | 0.555 | 0.465 | 0.046 | 0.122 | 0.022 | 3.34 | 0.0003 |
| **12** | **0.554** | **0.476** | 0.050 | 0.119 | 0.021 | 3.35 | 0 |

**Geometry plateau confirmed at matched_iou ≈ 0.55.** Doubling epochs from 6 (NB57) to 12 added only +0.001. The width-0.5 anchor head has hit its geometry ceiling on the full 70K dataset.

## The diagnosis at 29 experiments — two-headed problem

| metric | anchor head | DN-DETR head |
|---|---|---|
| Geometry | **stable**, ceiling 0.554 | unstable, oscillates 0.09-0.22 |
| Cls discrimination | **stable but uniform** (gap ≤ 0.02) | **unstable**, oscillates 0.22-0.62 |

The DN-DETR's cls signal is **REAL**: it produces 0.623 F1 at epoch 3 and 0.594 best_f1 at epoch 11. The architecture has enough capacity. The DAB anchor parameters drift between "good positions" (cls works at 60+ F1) and "bad positions" (cls fails to 25% F1). Classic DAB-DETR instability that published papers (Liu et al. ICLR 2022) note must be addressed by careful hyperparameter tuning or EMA weight averaging.

## Three new experiments

### Exp2FFF (NB61, exp56) — isolate joint conflict from DAB instability

Set `lambda_det=0.0` on the NB59 setup. Det loss still computed for logging but contributes zero to backward. If val_lane_f1 stabilizes at 0.5+ without oscillation, joint conflict was the cause of NB59's drift. If it still oscillates, DAB anchors are intrinsically unstable.

Single diff vs NB59: `lambda_det: 1.5 → 0.0`. Same long head_warmup (10 ep) + slow backbone (bb=0.02).

### Exp2GGG (NB62, exp57) — combine NB58's small cls win with NB60's geometry champion

`cls_separate_path=True` on the NB60 setup. NB58 showed disjoint cls path gives 5× improvement on 3K data but limited absolute level. Tests whether the cls_separate_path helps more at full data scale.

Single diff vs NB60 (exp55): `cls_separate_path: false → true`.

### Exp2HHH (NB63, exp58) — aggressive training-side stabilization of DAB-DETR

Brute-force stabilize DAB anchor parameters via reduced effective LR:
- `lr0: 0.0002 → 0.0001`
- `backbone_lr_mult: 0.02 → 0.005` (200× slower than baseline)
- `weight_decay: 0.0005 → 0.001`
- `grad_clip_norm: 5.0 → 2.0`
- `head_warmup until_epoch: 10 → 12`
- `lambda_det: 1.5` (slight boost prevents det collapse)

This is the published "patient DETR" recipe. If sustained val_lane_f1 ≥ 0.40 from epoch 10 onward, we have a stably-trained query head ready for fusion with anchor curves.

## Code state

No code changes needed. 3 new configs (`exp56`/`exp57`/`exp58`), 3 new notebooks (NB61/NB62/NB63), NB08 patched. `compileall` passes.

## Pass criteria

### Exp2FFF at epoch 20
- val_lane_f1 stays ≥ 0.45 from epoch 5 onwards (no oscillation = joint conflict was the cause)
- val_lane_best_f1 stays ≥ 0.50
- gap stays ≥ 0.15

### Exp2GGG at epoch 12
- val/matched_line_iou ≥ 0.55 (match NB60)
- val/lane_best_f1 ≥ 0.20 (2× NB60's 0.119)
- gap ≥ 0.05

### Exp2HHH at epoch 20 — the smoking gun
- **val_lane_f1 ≥ 0.40 SUSTAINED from epoch 10 onwards** (vs NB59's oscillation)
- matched_iou ≥ 0.20
- val_det ≤ 2.5 (no late collapse)

## The endgame view

After 29 experiments, the architectural answer is clear: **two-head ensemble**. NB60 gives the geometry (matched_iou=0.554), NB59 (or Exp2HHH if it stabilizes) gives the cls (val_lane_f1=0.62 at peak). Combined at inference, decoded_f1 should jump from current 0.062 to 0.20-0.35 — competitive with published joint-trained lane detectors.

Once Exp2HHH confirms cls stability, the next experiment (Exp2III) will be the **inference-time ensemble**: load anchor model from NB60's tar + load query model from Exp2HHH's tar + use anchor's coord_pred as the curves + use query head's per-prior cls re-rank applied via cross-attention onto the anchor's curve features. That's the ship-ready model.

If Exp2FFF shows det's gradient IS the cause of NB59's oscillation, then we know the path forward is: train DN-DETR lane-only first, then add det as a frozen-backbone fine-tune. The det branch is treated as a "downstream consumer" of the lane-trained backbone, not a co-equal task.

If Exp2GGG (cls_sep on full data) doesn't move beyond NB60: confirms anchor head is fundamentally cls-bottlenecked at this representation scale; the ensemble path becomes mandatory.

## Run order

1. **Exp2FFF first** (~35 min, 3K). Diagnostic; tells us the cause of NB59's oscillation.
2. **Exp2HHH** (~35 min, 3K). Stabilized cls model. Independent of Exp2FFF.
3. **Exp2GGG** (~2.5-3 hr, full 70K). Tests anchor cls_sep at scale.

Independent of each other and all prior NBs.
