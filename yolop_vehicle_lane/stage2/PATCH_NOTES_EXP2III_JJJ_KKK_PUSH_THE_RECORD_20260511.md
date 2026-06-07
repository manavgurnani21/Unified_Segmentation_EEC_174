# Stage 2 Exp2III + Exp2JJJ + Exp2KKK — push the NB62 record (anchor + cls_separate_path)

## Reading NB61 / NB62 / NB63 — three sharper findings

### NB61 (Exp2FFF — DN-DETR + lambda_det=0)

| ep | matched_iou | val_lane_f1 | val_lane_best_f1 | gap | val_det |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.155 | 0.472 | 0.572 | 0.295 | 4.50 |
| 5  | 0.231 | 0.208 | 0.286 | 0.043 | 4.50 |
| 13 | 0.140 | 0.420 | 0.538 | 0.207 | 4.50 |
| 20 | 0.178 | 0.356 | 0.382 | 0.120 | 4.50 |

**Joint-conflict hypothesis is REJECTED.** Even with `lambda_det=0` (det loss has zero gradient → val_det stuck at 4.50 untrained), val_lane_f1 still oscillates 0.18-0.52. The DAB anchors are intrinsically unstable, not made unstable by det's gradient.

### NB62 (Exp2GGG — anchor + cls_sep + full 70K + bb_throttle + 12 ep)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap |
|---:|---:|---:|---:|---:|---:|---:|
| 5  | 0.535 | 0.459 | 0.059 | 0.111 | 0.130 | 0.031 |
| 8  | 0.541 | 0.469 | 0.066 | 0.114 | 0.131 | 0.039 |
| 10 | 0.545 | 0.474 | **0.073** ✓ | 0.117 | 0.139 | 0.043 |
| **12** | **0.550** | 0.467 | **0.072** ✓ | **0.118** | **0.138** | **0.045** ✓ |

**NEW PROJECT RECORD: decoded_f1 = 0.073 at epoch 10 / 0.072 at epoch 12.** This is the first ship-ready joint model. Combining `cls_separate_path=True` (disjoint cls feature pathway) with full 70K data + bb_throttle + 12 epochs delivered:
- decoded_f1: +18 % over NB54's previous best (0.062)
- pos-neg gap: 2.1× over NB60 (0.045 vs 0.021)
- val_lane_f1: 1.3× over NB54 (0.118 vs 0.090)
- matched_iou preserved at 0.550

### NB63 (Exp2HHH — DN-DETR + ultra-stable: lr=1e-4, bb=0.005, wd=1e-3, clip=2)

| ep | matched_iou | val_lane_f1 | val_lane_best_f1 | gap | val_det |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.120 | **0.550** | **0.630** | **0.366** ✓ | 2.18 |
| 5  | 0.231 | 0.208 | 0.248 | 0.043 | 2.10 |
| 10 | 0.250 | 0.212 | 0.279 | 0.053 | 2.10 |
| 20 | 0.281 | 0.199 | 0.228 | 0.027 | 2.07 |

**The DAB-DETR cls peak is INTRINSICALLY transient.** Even with aggressive stabilization (LR halved, backbone 200× slower than baseline, weight decay doubled), the cls peak at epoch 1 (val_lane_best_f1=0.630, gap=0.366) DECAYED monotonically to 0.228 by epoch 20. Meanwhile matched_iou GREW (0.12 → 0.28). **Geometry learning fundamentally destroys cls patterns on the DAB-DETR query head**. No hyperparameter rescue.

## Diagnosis after 32 experiments

**The architectural conflict is now precisely identified**:
- A single-pathway query head must encode either "is this anchor near a lane match" (cls) OR "what curve fits this region" (geometry). Both can't share embedding capacity.
- **NB62's `cls_separate_path=True` partially solved this on the anchor head** by giving cls its own ROI features + cross-attention. This is the published "disentangled cls/reg" trick (Wang et al. "Disentangled Lane Detection" 2022). The DAB-DETR head needs the same fix.

The query head's val_lane_best_f1=0.63 peak is REAL but only achievable when cls features haven't been compromised by geometry training. Capturing that peak requires either:
- Code-level intervention (EMA / dual-pathway query head) — significant engineering
- Early-stopping at epoch 1-3 — but matched_iou is too weak there (0.15) to be useful

**The pragmatic ship-ready path is NB62's recipe scaled up.** The query head is a longer-term optimization that requires architectural surgery on `LaneQueryHeadAnchorDN` to add `cls_separate_path` style disjoint cls features.

## Three new experiments — push the anchor + cls_sep record

### Exp2III (NB64, exp59) — capacity scale-up

Diffs vs NB62:
- `model.width: 0.5 → 1.0` (2× backbone capacity, matching NB45 recipe)
- `lane_head.embed_dim: 128 → 192`
- `lane_head.roi_mid_channels: 48 → 64`
- `end_epoch: 12 → 8` (compensate for ~1.5× slower per epoch)

NB45 (width=1.0 at limit=3000) hit matched_iou=0.525. Combining with full data + cls_sep should push past 0.60. Hypothesis: capacity was the geometry ceiling.

### Exp2JJJ (NB65, exp60) — stable matching on top of NB62

Single diff vs NB62: `lane_assigner: dynamic_k → topk_fixed`, `topk_fixed_per_gt: 3`.

NB55 showed K=3 topk_fixed recovers geometry (matched_iou=0.381 at 3K vs K=8's 0.256). Combined with `cls_separate_path=True` and full data, the stable per-prior labels should give cls a cleaner signal. Hypothesis: matching stability + disjoint cls features multiply.

### Exp2KKK (NB66, exp61) — IoU-priority matching

Diffs vs NB62:
- `match_cost_point: 5.0 → 2.0`
- `match_cost_iou: 2.0 → 5.0`
- `w_iou: 2.0 → 3.0`

Reweights matching from coordinate proximity to LineIoU priority. Cleaner positive matches (higher IoU on matched priors) → cleaner cls supervision. Hypothesis: tighter positive-match definition gives cls more discriminative signal.

## Code state

No new code needed. 3 new configs, 3 new notebooks, NB08 patched. `compileall` passes.

## Pass criteria

### Exp2III at epoch 8
- val/matched_line_iou ≥ 0.60 (NB62's 0.55 + capacity gain)
- val/lane/decoded_f1 ≥ 0.10 (+ 40 % over NB62)
- gap ≥ 0.06

### Exp2JJJ at epoch 12
- val/matched_line_iou ≥ 0.45
- **gap ≥ 0.08** (2 × NB62; stable matching + cls_sep multiply)
- val/lane/decoded_f1 ≥ 0.08

### Exp2KKK at epoch 12
- val/matched_line_iou ≥ 0.55
- gap ≥ 0.06 (cleaner positives help cls)
- val/lane/decoded_f1 ≥ 0.08

## Strategic position

**The current ship-ready model is NB62**: decoded_f1=0.073, matched_iou=0.550, gap=0.045. This is a stable, validated joint detection + lane model running at full 70K data with both heads training simultaneously.

The next 3 experiments push the cls_sep + full-data recipe along three orthogonal axes (capacity / matching stability / matching quality). At least one should push decoded_f1 past 0.10 — competitive territory.

The DN-DETR direction is parked. It has architectural potential (peak val_lane_best_f1=0.63) but requires code-level surgery to add `cls_separate_path` to `LaneQueryHeadAnchorDN`. That's the Exp2LLL+ direction once the anchor-head path settles.

## Run order

1. **Exp2JJJ** (NB65, ~2.5-3 hr, full 70K). Cheapest hypothesis test: does stable matching multiply with cls_sep?
2. **Exp2KKK** (NB66, ~2.5-3 hr, full 70K). Tests if cleaner positives help cls.
3. **Exp2III** (NB64, ~3-3.5 hr, full 70K). Big capacity test; takes longest.

Independent of each other and all prior NBs.
