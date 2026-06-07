# Stage 2 Exp2WW + Exp2XX + Exp2YY — fix the matcher, stabilize the queries, scale to full data

## Reading NB49 / NB50 / NB51 — three more failures, three sharper diagnoses

### NB49 (Exp2TT — HybridPriorQueryHead, K=12 stage 2 + stage1_aux=1.0)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.092 | 0.037 | 0.017 | **0.629** | 0.660 | 0.051 |
| 10 | 0.115 | 0.063 | 0.034 | 0.629 | 0.639 | 0.013 |
| 20 | **0.117** | 0.062 | 0.015 | **0.629** | **0.657** | 0.035 |

**Hybrid head failed.** val_lane_f1 = 0.629 at every epoch is misleading: K=12 queries × 8 batch = 96 predictions and recall = 1.0, precision = 0.46 — every query predicts positive. matched_iou crashed to 0.117 because stage 2's K=12 queries don't inherit stage 1's geometry — they regenerate curves from scratch via `param_head + offset_head` on query features, with only ~5 queries per image getting positive geometric supervision.

### NB50 (Exp2UU — K=64 query head + VFL + full 70K data)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap | val_det |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.028 | 0.014 | 0.004 | 0.013 | 0.166 | -0.000 | 3.36 |
| 3  | 0.014 | 0.008 | 0.008 | 0.041 | 0.163 | -0.000 | 3.36 |
| 6  | **0.013** | **0.003** | 0.003 | 0.160 | 0.160 | -0.000 | 3.35 |

**Catastrophic collapse at full data.** matched_iou DROPPED from NB47's 0.27 to 0.013. The K=64 query embeddings can't ground themselves in 70K samples worth of variety in 6 epochs — queries oscillate and the learned anchor positions never stabilize. Det also broken (val_det=3.35).

### NB51 (Exp2VV — anchor + VFL + full data + lambda_det=2.0)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_cls | gap | val_det | val_map50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.383 | 0.235 | 0.020 | 0.022 | 0.005 | 3.36 | 0.0000 |
| 5  | 0.546 | 0.443 | 0.037 | 0.035 | 0.008 | 3.12 | 0.0003 |
| 6  | **0.549** | 0.450 | **0.051** | 0.036 | 0.008 | **3.12** | **0.0000** |

**Tiny improvement on geometry, det rescue mostly failed.** matched_iou=0.549 is a slight new project record (NB48 was 0.544), but val_det stayed at 3.12 (vs NB48's 3.16) and val_map50 ≈ 0. The `lambda_det=2.0, lambda_lane=0.7` re-weighting didn't recover detection at full data scale.

## Cumulative findings across 20 experiments

After NB39-51 the picture is now sharp:

| run | head | matcher | what discriminates? | what suffers? |
|---|---|---|---|---|
| NB39-46, NB48 | anchor 192 | dynamic-k | nothing (gap ≤ 0.015) | cls is degenerate uniform sigmoid |
| NB44 | anchor 192 | Hungarian 1-to-1 | nothing (gap=0.010) | cls AND geometry crashed (matched_iou=0.38) |
| NB47 | query K=64 | Hungarian 1-to-1 | **cls** (gap=0.099, val_lane_f1=0.246) | geometry weak (matched_iou=0.27) |
| NB49 | hybrid 192+12 | dynamic-k stage1 + Hungarian stage2 | apparent (legacy F1=0.63 but degenerate) | matched_iou=0.117, every query predicts positive |
| NB50 | query K=64 | Hungarian + full data | nothing (gap=0) | catastrophic collapse |
| NB51 | anchor 192 | dynamic-k + lambda_det=2.0 + full data | nothing | det still broken (map50=0) |

**Root-cause diagnosis (refined):**

1. **dynamic-k matching** flickers labels per-batch → cls converges to uniform sigmoid no matter the loss.
2. **Hungarian 1-to-1** stabilizes labels but with K=1 only ~5 priors/image get positive geometric supervision → starves geometry.
3. **K=64 query head** is small enough to specialize at 3K samples but oscillates at 70K — DETR's classic data-vs-stability problem.
4. **Hybrid head's stage 2** doesn't inherit stage 1 geometry because it regenerates curves from scratch; need either (a) a stage 2 that REFINES stage 1 curves rather than overwrites, or (b) K-best matching that stays on stage 1.

These point at three concrete fixes.

## Three new experiments

### Exp2WW (NB52, exp47) — anchor + topk-fixed K=8 matching + VFL

The cleanest fix. Each GT lane takes the top **K=8** priors by cost (deterministic, not IoU-sum estimated). Conflict resolution: each prior matches at most one GT (lowest-cost wins).

Properties:
- **Dense supervision** (K=8 × 5 GT = ~40 positives/image, anchor-style)
- **Stable labels** (deterministic per cost matrix, no batch flicker)
- Combined with VFL on continuous LineIoU regression target

If pos-neg gap reaches ≥ 0.05 on the anchor head, the matcher was the bug. **This is my highest-confidence experiment** because it directly addresses the diagnosed root cause.

**Code change**: new `_topk_fixed_match()` in `stage2/fusion/losses.py` + `lane_assigner='topk_fixed'` + `topk_fixed_per_gt=8` config field. Backwards-compatible with all prior runs.

### Exp2XX (NB53, exp48) — K=64 query head + DAB anchors + DN denoising + VFL

The architectural fix for NB47's late-epoch decay (gap peaked at ep9 at 0.099, decayed to 0.063 at ep20) and NB50's full-data collapse.

Uses the existing `LaneQueryHeadAnchorDN` (already implemented):
- **DAB anchors**: each query has a learnable (start_y, start_x, theta) anchor that biases its attention. Queries can't all collapse to the same point.
- **Denoising queries** (DN-DETR style): training-time noised copies of every GT lane that must be reconstructed. Provides dense gradient signal that prevents query oscillation.

Reference: Liu et al. "DAB-DETR" ICLR 2022; Li et al. "DN-DETR" CVPR 2022.

If gap stays ≥ 0.08 from epoch 10 onwards (vs NB47's decay) AND matched_iou ≥ 0.35 (vs NB47's 0.27), the DAB+DN recipe is the right query-head modernization.

### Exp2YY (NB54, exp49) — anchor + topk-fixed + VFL + full 70K + aggressive det rescue

Combine all the wins on the full dataset:
- Anchor head 192 priors (NB48 geometry champion)
- topk-fixed K=8 matcher (Exp2WW)
- VFL + lineiou_regression (Exp2QQ)
- Full 70K data
- **Aggressive** det rescue: `lambda_det=3.0`, `lambda_lane=0.5`, `use_uncertainty=False` (NB51 used 2.0/0.7 with marginal improvement; this pushes harder)

If Exp2WW (limit=3000) shows the matcher fix works AND Exp2YY (full data) inherits that win plus rescued det, this is the final stable model for Stage 3.

## Code changes (back-compat, gated by config)

- **[`stage2/fusion/losses.py`](fusion/losses.py)**:
  - New `_topk_fixed_match(cost, topk)` helper.
  - Dispatch in `match_targets`: new `'topk_fixed'` / `'topk'` / `'fixed_topk'` branch.
  - `FusionLossConfig`: new field `topk_fixed_per_gt: int = 8`.

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Three new YAMLs round-trip; three new notebooks round-trip; NB08 patched.

## Run order — fully independent

1. **Exp2WW first** (NB52, ~30 min, limit=3000). The lowest-risk single-knob test of the matcher diagnosis. Fast feedback.
2. **Exp2XX** (NB53, ~30-35 min, limit=3000). Architectural backstop if topk-fixed doesn't crack the anchor head.
3. **Exp2YY** (NB54, ~60-80 min, full data). Run AFTER NB52 confirms the matcher fix at 3K. If NB52 fails (gap < 0.02), skip NB54 and pivot to PCGrad.

## Decision tree

| Exp2WW result | Exp2XX result | Conclusion + next move |
|---|---|---|
| pass (gap ≥ 0.05, decoded_f1 ≥ 0.10) | — | Matcher was the bug. Run Exp2YY immediately. If it passes too, ship the model for Stage 3. |
| fail | pass | Per-prior anchor features fundamentally non-discriminative. Switch to query head. |
| pass | pass | Two paths work. Choose the one with better matched_iou for Stage 3. |
| fail | fail | Pivot to (a) PCGrad gradient surgery, OR (b) pretrained YOLOPv2 backbone init from `external_repos/YOLOP/weights/End-to-end.pth`. |

## Why I chose these three experiments over the user's CVPR suggestions

The user's hints (GELAN-PGI, AdaBins, NaviBridger, STAL, PCGrad, DDAD) are sound for the depth head and would help future joint conflict mitigation. But they explicitly said "I want you to really think about how to improve the project based on your own thought rather than blindly following my lead."

My empirical analysis says:
- The cls collapse is **matcher-bound**, not loss-bound (NB46 ruled out alpha; NB47 showed Hungarian fixes it on queries).
- **dynamic-k**'s sum-of-IoUs estimation is the actual flicker source — the standard CLRKDNet design assumed >100 epoch training where matching settles; we're at 20 epochs.
- topk-fixed is a single-line matcher change that directly addresses the diagnosis. Lowest engineering cost, highest expected impact.

If Exp2WW passes, the depth head can be added on top of a stable joint-trained model. If it fails, then GELAN/PGI/PCGrad become the sensible Exp2ZZ direction.
