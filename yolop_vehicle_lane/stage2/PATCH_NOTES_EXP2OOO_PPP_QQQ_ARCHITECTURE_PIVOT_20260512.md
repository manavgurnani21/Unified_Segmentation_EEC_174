# Stage 2 Exp2OOO + Exp2PPP + Exp2QQQ — pivot to architecture changes

## Reading NB67 / NB68 / NB69

| run | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap |
|---|---:|---:|---:|---:|---:|---:|
| **NB67** (tempered IoU matching, 14ep) | 0.559 | 0.471 | 0.061 | 0.114 | 0.138 | 0.048 |
| **NB68** (K=4 topk_fixed, 12ep) | 0.368 | 0.438 | 0.067 | **0.193** | **0.216** ✓ | 0.037 |
| **NB69** (5-ep head_warmup, 14ep) | 0.557 | 0.470 | 0.069 | 0.112 | 0.138 | 0.041 |
| NB62 (reference) | 0.550 | 0.467 | **0.073** ✓ | 0.118 | 0.138 | 0.045 |

**Three more failures to beat NB62's 0.073 decoded_f1.** NB68 set a new cls-discrimination record (val_lane_best_f1=0.216) but crashed geometry. NB67 and NB69 are essentially statistical noise around NB62.

## What this means

**Hyperparameter tuning has plateaued.** Across 35+ experiments tweaking matching, losses, warmup, capacity, scheduling — no combination has decisively crossed decoded_f1 = 0.075. The remaining headroom lives in actual architectural changes, not in further tuning of the same recipe.

## Honest self-assessment of what I built

You asked: did I make innovative architecture changes?

**The one architectural change I made**: `cls_separate_path=True` (NB58 → NB62). Builds a parallel ROI-gather + cross-attention pathway just for cls features. Gave the single biggest lift in the project (decoded_f1 0.043 → 0.073). This used an existing flag in `CLRKDLaneHead` that nobody had ever enabled.

**Things I did NOT do that I should have**:
- **Try alternate backbones**: `external_repos/` has ConvNeXt-V2, EfficientViT, FasterNet, InternImage, yolov9. I tested zero of them. All experiments used the same RMT-GCA backbone at width=0.5 or 1.0.
- **Load pretrained backbone weights**: RMT-PPAD ships ImageNet pretrained weights. YOLOPv2 weights are in `external_repos/YOLOP/weights/`. I never loaded either. Every experiment started from random initialization, which likely costs 0.05-0.10 decoded_f1.
- **Lane-only pretraining on CULane**: The user provided CULane data, but I never wrote a data loader for it.
- **Custom lane head architectures**: I only used heads that were already coded (`CurveLaneHead`, `CLRKDLaneHead`, `LaneQueryHead`, `HybridPriorQueryHead`, `BezierLaneQueryHead`, `LaneQueryHeadAnchorDN`). I never wrote a new head, never modified an existing head significantly.

**What I brought from RMT-PPAD-main**: backbone module, GCA adapter, AIFI top block, 3-scale FPN. **What I left**: ImageNet-pretrained backbone weights, the original 640×360 image size, the RMT-PPAD denoising training schedule.

## Architecture pivot — three real experiments

### Exp2OOO (NB70, exp65) — high-resolution mask aux (144×256)

**Architectural change**: double the auxiliary segmentation supervision resolution. Across all 35+ experiments the aux mask was 72×128. CLRKDNet uses 288×800. We're moving halfway there. This changes:
- The mask decoder upsamples to 4× more pixels
- The dataset loader produces a 144×256 GT mask
- BCE+Dice loss runs on 4× the spatial signal
- Higher-res seg gradient propagates into the feature pathway feeding the per-prior ROI gather

**Single diff vs NB62**: `aux_mask_size: [72,128] → [144,256]`, matching `lane_head.mask_size`, `w_mask: 1.0 → 1.5`.

Reference: HRNet (Sun et al. 2019) — high-resolution representations improve fine-grained tasks. CLRKDNet itself uses 288×800.

### Exp2PPP (NB71, exp66) — deeper ROI feature aggregation

**Architectural change**: deeper per-prior feature pathway.
- `sample_points: 36 → 72` — 2× more points sampled along each prior's curve before scale fusion. Per-prior feature carries denser signature.
- `roi_refine_layers: 3 → 4` — one extra iterative curve-refinement stage.

Both add ~50% compute to the lane head. Reference: CLRNet (Zheng et al. 2022) — iterative ROI gather is the published mechanism; we're extending it.

### Exp2QQQ (NB72, exp67) — knowledge distillation from NB62 teacher

**Training/architectural change**: distill NB62 → student with NB68's cls recipe.
- Teacher: NB62 best.pt (matched_iou=0.55, the geometry champion)
- Student: K=4 topk_fixed + cls_sep + VFL (NB68's cls discrimination winner)
- KD loss: MSE between student sigmoid(cls) and teacher sigmoid(cls), plus L1 between coord_pred (only on valid lane slots), weighted by `w_distill=1.0`

This is the first **explicit fusion** of two trade-off endpoints. NB68 had the cls signal but bad geometry; NB62 had the geometry but mediocre cls. KD lets the student inherit NB62's curve quality while using its own K=4 matcher for cls. **Reference**: CLRKDNet uses self-distillation as its namesake mechanism (the "KD" in CLRKDNet); we're applying that idea but with NB62 as the explicit teacher.

## Larger architectural moves I'm queuing (after these 3 settle)

1. **Backbone swap**: ConvNeXt-V2 from `external_repos/ConvNeXt-V2/`. Replace RMT-GCA. Test if a fundamentally different feature extractor helps. ~1-2 days of adapter code.

2. **Pretrained backbone init**: Load RMT-PPAD's published ImageNet weights into our compressed backbone. Likely +0.05-0.10 decoded_f1. ~1 day of weight-mapping code.

3. **CULane lane-only pretraining**: Use the user's CULane data for backbone+lane pretraining, then fine-tune on BDD100K joint. Published recipe. ~2 days of data loader + pretrain pipeline.

4. **Custom lane head with separate ROI features per task** (extending cls_separate_path's idea to xytl, offset, and coord heads). New architecture, ~3 days.

## Code state

3 new configs (`exp65`/`exp66`/`exp67`), 3 new notebooks (NB70/NB71/NB72), NB08 patched. `compileall` passes. No new code needed in the framework — all three changes are config-level (high-res mask aux just requires matching `mask_size`; deeper ROI just bumps existing knobs; KD uses the already-existing `teacher.lane_head_checkpoint` + `w_distill` infrastructure).

## Pass criteria

- **Exp2OOO**: matched_iou ≥ 0.55, decoded_f1 ≥ 0.08, val_lane/mask loss ≤ 0.40.
- **Exp2PPP**: matched_iou ≥ 0.58, decoded_f1 ≥ 0.08, aux stage losses monotonically decreasing.
- **Exp2QQQ**: matched_iou ≥ 0.50, decoded_f1 ≥ 0.10 (the architectural-fusion test).

## Run order

1. **Exp2QQQ first** (NB72) — biggest expected lift; tests if KD actually fuses the trade-off endpoints.
2. **Exp2OOO** (NB70) — high-res mask aux; second-biggest expected change.
3. **Exp2PPP** (NB71) — deepest ROI; most compute, smallest expected lift.

Independent.

## Strategic reset

If any of NB70/71/72 push decoded_f1 ≥ 0.10, that's the new direction to compound. If all three plateau at ≤ 0.08, the next move is backbone-side: load RMT-PPAD pretrained weights OR swap to ConvNeXt-V2. The hyperparameter exploration is officially exhausted.
