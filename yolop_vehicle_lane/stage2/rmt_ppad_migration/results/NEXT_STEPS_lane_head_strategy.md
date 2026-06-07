# Lane-Head Next-Steps Strategy — Breaking the Plateau, Fixing Shape & Cardinality

**Status:** strategy / analysis only. No code is changed by this document.
**Scope:** the lane branch of the `combined_aux_gca_full` model (NB101), evaluated qualitatively by NB103.
**Author's note:** every claim below is grounded in the actual training CSVs, the loss/metric/decode code, and the NB103 viewer. Sources are cited inline by path.

---

## 0. Executive summary (read this first)

You report three things: (1) lane F1 and the other lane metrics have flatlined for the last ~5 epochs; (2) predicted lanes roughly align with GT but the **near-camera, center lanes are jagged / sawtooth / swaying** while far lanes are smooth; (3) the model **always emits 8 lanes** even when GT has ~0.

After reading `combined_aux_gca_full/results.csv`, the loss (`P5_loss/tools/lane_losses.py`), the metric/decoder (`P7_validator/tools/lane_curve_f1.py`), and the NB103 inference helper, the diagnosis is that **these are not three problems — they are three faces of two root causes**:

- **Root cause #1 — representation.** The lane head is the CLRNet 78-D polyline: it regresses **72 x-offsets at fixed y-rows, independently, with no inter-row smoothness prior**, and the LineIoU loss uses a thick tolerance band that does not punish wiggle. Far-field (near-vertical, slowly-varying x) looks smooth; near-field (x changes fast, so the same angular noise becomes large per-row x noise) becomes a sawtooth. This representation also caps geometric quality, which is why curveIoU sits at ~0.09 and F1 ceilings at ~0.236.
- **Root cause #2 — set prediction / cardinality.** The foreground classifier never learns a confident positive/negative separation (`lane_score_std ≈ 0.09`, the loss code itself documents a pos-neg gap of only ~0.099 after the Hungarian fix), AND the NB103 decode takes **`top-k = 8` priors with no confidence threshold** (boxes are thresholded at conf > 0.3; lanes are not). So the model structurally cannot say "there are 0 lanes here."

**A critical measurement caveat:** the Bezier runs (`bezier_cubic_no_lcm`, `bezier_lcm_gamma001`) show `lane_f1 = 0` for all 30 epochs — but the Bezier branch of the metric (`_decode_lane_bezier`) **gates predictions on `row[1] < 0.5 → None`**, which is exactly the bug the polyline decoder warns against (pred scores are routinely < 0.5). So the Bezier experiments were very likely **mis-scored and abandoned prematurely**. They must be re-scored with a fixed decoder before we conclude curves don't work — because a smooth curve representation is the single most direct fix for the jaggedness.

**The meta-insight:** you have exhausted the *negative-transfer / "share-better"* axis (NB94-101 ablations: aux_seg + gca_floor won, mAP50 holds ~0.50, the trunk is healthy). The remaining gains are **inside the lane head** — its geometry representation and its set-prediction calibration. That is where this plan focuses.

The plan is four sprints with hard decision gates:
- **Sprint 0 (instrument, ~2 days, no training):** prove the cheap hypotheses before spending GPU.
- **Sprint 1 (cheap high-yield, ~1 week):** decode threshold + smoothness reg + y-reweight + IoU-aware cls target. Small deltas to the current winner.
- **Sprint 2 (representation, ~2-3 weeks):** CLRerNet LaneIoU recipe and/or a re-scored Bezier head.
- **Sprint 3 (bold, ~3-5 weeks):** a curve-query DETR head that unifies lanes with the existing RT-DETR detection decoder (MapTR-style matching), or a GANet keypoint head.

---

## 1. Evidence digest

### 1.1 The plateau is real and specific (NB101 = `combined_aux_gca_full/results.csv`)

| epoch | lane_f1 | curveIoU | pixIoU | mAP50 | startx_std | length_mean | score_std |
|---|---|---|---|---|---|---|---|
| 5 | 0.124 | 0.079 | 0.118 | 0.445 | 0.047 | 0.315 | 0.079 |
| 8 | 0.228 | 0.083 | 0.104 | 0.477 | 0.042 | 0.314 | 0.083 |
| 9 | **0.244** | 0.086 | 0.110 | 0.483 | 0.046 | 0.294 | 0.086 |
| 12 | 0.236 | 0.088 | 0.111 | 0.493 | 0.044 | 0.287 | 0.088 |
| 16 | 0.237 | 0.091 | 0.112 | 0.499 | 0.042 | 0.282 | 0.091 |

Readings:
- **lane_f1 peaks at epoch 9 (0.244) and is flat (0.236-0.237) through 16** → exactly the "stopped rising in the last 5 epochs" you described.
- **curveIoU is near-saturated at ~0.09** → the geometry itself is coarse; F1 is bounded by it.
- **mAP50 holds ~0.50** → detection is healthy; this is a lane-head problem, not an MTL problem.
- **`lane_startx_std ≈ 0.04` (tiny)** → all predicted lanes start at nearly the same x (~center). The priors are collapsing to center-bottom. This is the quantitative fingerprint of your "center lane is the worst, others vary" observation.
- **`lane_length_mean` DECREASES 0.38 → 0.28 over training** → the model is learning to predict *shorter* lanes. Hypothesis: LineIoU's tolerance band lets it cover the easy near-vertical top and *give up on the hard near-field bottom*, where the jaggedness lives.
- **`lane_score_std ≈ 0.08-0.09`** → the classifier never sharpens; positive and negative priors stay nearly indistinguishable in score.

### 1.2 The decode cannot produce a variable lane count (NB103, `nb103_build_helper.py`)

```python
# infer(): lane decode
order = np.argsort(-lcf1._softmax_pos(rows))[:lane_topk]   # lane_topk = 8
for r in rows[order]:
    poly = lcf1._decode_row(...)        # renders ALL of them, no score gate
```
Boxes get `conf > 0.3`; lanes get **no threshold at all**. So every image renders 8 lanes by construction. Even the validator's F1 path (`lane_curve_tp_fp_fn`) takes the top-`max_lanes` by score with no absolute cutoff. **There is no mechanism anywhere that can output "0 lanes."**

### 1.3 The classifier collapse is documented in your own code (`lane_losses.py`)

The `hungarian_assign` docstring:
> "dynamic-k labels the same prior positive in some batches / negative in others, so the binary cls converges to a uniform 0.5 sigmoid … Hungarian gives every matched prior a stable positive label … (Exp2RR/NB47 broke an 11-experiment plateau exactly this way: pos-neg gap 0.01 → 0.099)."

So you already fought this once and moved the gap from 0.01 to ~0.099. **0.099 is still tiny** — it explains why no fixed threshold can cleanly separate 8→variable. The matching fix was necessary but not sufficient.

### 1.4 The Bezier runs are probably mis-measured (`lane_curve_f1.py`)

```python
def _decode_lane_bezier(row, R, num_points=72):
    if float(row[1]) < 0.5:      # <-- gates PREDICTIONS on score >= 0.5
        return None
```
Compare the polyline path, which deliberately does NOT gate preds:
> "PRED row[1] is a raw cls LOGIT that is routinely < 0.5 even for a good-geometry prior, so gating preds on it drops almost every predicted lane → curveIoU pinned at 0."

The Bezier decoder does the exact thing the polyline decoder warns against. Result: `bezier_*/results.csv` shows `lane_f1 = 0` for all 30 epochs while `lane_bezier_span ≈ 0.98` (the curves do span the image). **The Bezier head may have fine geometry that the metric zeroed out.** This must be settled in Sprint 0.

---

## 2. Root-cause analysis

```
  SYMPTOM                              ROOT CAUSE                         FIX FAMILY
  ───────────────────────────────────────────────────────────────────────────────
  near-field jagged / sawtooth   ┐
  center lane worst              ├──► R1: per-row independent x-offset ──► smoothness prior /
  length shrinking over training ┘     polyline + tolerant LineIoU         low-DOF curve / IoU reweight
                                        (no global shape prior)

  F1 ceiling ~0.236              ┐
  curveIoU stuck ~0.09           ├──► R1 (coarse geometry) + R2 (cls    ──► better IoU loss & matching
                                 ┘     bottleneck on ranking)              (CLRerNet) / new representation

  always 8 lanes                 ┐
  8 lanes when GT≈0              ├──► R2: weak/uncalibrated foreground  ──► threshold decode + IoU-aware
  precision << recall           ┘     cls + threshold-free top-k decode    cls target + existence head /
                                        + extreme pos/neg imbalance         one-to-one set prediction
```

Two cliffs, not six. Everything below attacks R1 (geometry/shape) and R2 (cardinality/calibration).

---

## 3. The reference shelf (what we already have, what to download)

Already in `external_repos/` — use directly:
- **CLRNet** — the anchor-line base; read its head + LineIoU + (importantly) its segmentation auxiliary.
- **CLRKDNet-master** — the vendored variant currently in use.
- **MapTR** — ⭐ the template for *variable-cardinality vectorized geometry via DETR queries* with permutation-invariant point matching. Directly relevant to both R2 (cardinality) and R1 (shape as a point set). Study `assigners` + `losses`.
- **YOLOP / YOLOPv2** — mask-based lane baselines (sanity reference for pixel IoU).
- Backbones (ConvNeXt-V2, FasterNet, InternImage, efficientvit) — optional trunk swaps later.
- `2208.11434v1.pdf` — verify which paper this is; keep if lane-relevant.

To download into `external_repos/` (each justified):
- **CLRerNet** (WACV 2024, `hirotomusiker/CLRerNet`) — ⭐ **highest priority**. Introduces **LaneIoU** (an IoU that accounts for local lane angle so the overlap measure is geometry-faithful) used for *both* assignment and loss, plus a **confidence calibration** that makes the score reflect geometric quality. This is the canonical published fix for "precision << recall + flat scores + plateaued IoU" — i.e. your exact symptom set, in the same model family. Minimal architecture change.
- **BezierLaneNet** (CVPR 2022, `aliyun/conditional-lane-detection` has a variant; canonical `Zhengtq/BezierLaneNet` or `voldemortX/pytorch-auto-drive`) — smooth-by-construction cubic Bezier head + a **feature-flip fusion** for the center symmetry. Direct fix for jaggedness and for the *center-lane* instability specifically.
- **GANet** (CVPR 2022, `Wolfwjs/GANet`) — keypoint + global association; cardinality is emergent (no fixed prior count), local keypoint regression is naturally smooth, strong on curved/dense scenes. A paradigm alternative that hits all three symptoms.
- **CondLaneNet** (ICCV 2021, `aliyun/conditional-lane-detection`) — row-wise + conditional convolution + an explicit **lane-existence / instance count** mechanism; good template for the cardinality head.
- (optional) **ADNet** (ICCV 2023) — flexible anchor *start points*, which targets your center-collapse `startx_std≈0.04` directly; **Sparse Laneformer** (2023) — sparse learnable lane queries (a lighter curve-query design than full MapTR).
- (optional) **pytorch-auto-drive** (`voldemortX/pytorch-auto-drive`) — one repo with clean re-impls of LSTR/BezierLaneNet/PolyLaneNet; convenient for borrowing the smooth-parameterization heads and their CULane/curve metrics.

> Suggested action for the next working session: `git clone` CLRerNet, GANet, BezierLaneNet (or pytorch-auto-drive), and CondLaneNet into `external_repos/`, plus pull their papers' PDFs. Keep them read-only references like the existing CLRNet/CLRKDNet.

---

## 4. Sprint 0 — Instrument & confirm (≈2 days, NO training)

Goal: prove the cheap hypotheses and remove measurement noise before spending GPU. Every item is a diagnostic notebook, not a training run.

**S0.1 — Fix the Bezier metric decoder and re-score the existing Bezier checkpoints.**
The single highest-information action. Patch `_decode_lane_bezier` to NOT gate preds on `row[1] < 0.5` (mirror the polyline path: rank by softmax score upstream, render geometry regardless), then re-run the curve-F1 metric over `bezier_cubic_no_lcm/best.pt` and `bezier_lcm_gamma001/best.pt`. *Hypothesis:* Bezier F1 jumps from a spurious 0 to something competitive, and its curves are visibly smoother than the polyline head. *If true,* the jaggedness fix is half-built already.

**S0.2 — Score-histogram diagnostic.** For NB101's best.pt over ~200 val images, plot the histogram of `softmax_pos` for matched (positive) vs unmatched (negative) priors. Quantify the separation (e.g. AUC, or the gap between the 8th-highest score and the 9th). *This tells you whether a threshold can ever work, and what τ would be.*

**S0.3 — Threshold sweep on the existing model (decode-only, no retrain).** Re-decode NB101 val predictions with `score > τ` (cap at max_lanes) for τ ∈ {0.1 … 0.9}; plot precision/recall/F1 and **predicted-lane-count vs GT-lane-count correlation** vs τ. *This quantifies how much of the "always 8" problem is pure decode (fixable today) vs calibration (needs retraining).*

**S0.4 — Near-vs-far error decomposition.** Split each predicted lane into top-half (far) and bottom-half (near) y-ranges; compute per-row x-error and a "jaggedness" score (mean squared 2nd difference of x along the lane) for each half, pred vs GT. *Confirms quantitatively that the wiggle is concentrated near-field and gives a number to optimize against.*

**S0.5 — Re-read NB95 (`lane_overfit_probe`) and NB96 (`geometry_lever_ablation`).** These existing diagnostics may already isolate "can the head fit a single image perfectly?" (capacity) vs "does it generalize?" (optimization). Fold their conclusions in before designing Sprint 1.

**Exit criterion for Sprint 0:** you can state, with numbers, (a) the true Bezier-head F1, (b) the best achievable decode-threshold F1 on the current model, (c) the near/far jaggedness ratio, (d) whether the cls is threshold-able. These four numbers decide how much of Sprint 1 is even needed.

---

## 5. Sprint 1 — Cheap, high-yield deltas to the current winner (≈1 week)

All four are *small, composable changes* to the existing polyline head + loss + decode. None is architecturally risky. Run them as an ablation grid on the 10k subset first (fast), promote winners to 70k.

### S1.1 — Confidence-thresholded decode (R2; do this first, it's free)
Replace `top-k = 8, no threshold` with `keep score > τ, then cap at max_lanes`, τ chosen from S0.3. Also expose τ as an eval arg so NB103 and the validator agree. *Expected:* immediate collapse of the "8 lanes when GT≈0" failure mode, and precision↑. *Risk:* near-zero; it is a decode change. *Caveat:* bounded by cls separation — if S0.2 shows no separation, this only partly helps and S1.3 becomes essential.

### S1.2 — Smoothness / curvature regularizer on the polyline (R1)
Add `L_smooth = mean_i (x_{i+1} - 2 x_i + x_{i-1})^2` over the valid offsets of each matched lane, **weighted more on near-field rows** (where wiggle lives). This is a 2nd-difference (discrete curvature) penalty — standard in spline fitting. *Expected:* the sawtooth visibly relaxes; near-field x-error variance drops; possibly a small curveIoU gain. *Risk:* low; over-weighting can over-smooth real curves — sweep the weight {0.01, 0.05, 0.2}. *Grounding:* the curvature prior is the discrete analogue of what Bezier/spline heads get for free.

### S1.3 — IoU-aware (soft) classification target (R2; the CLRerNet idea, cheap version)
Today the matched prior's cls target is a hard 1. Instead set the **positive target to the line-IoU between the matched prior and its GT** (a soft label in [0,1]), as CLRerNet/Quality-Focal-Loss do. The score then *encodes geometric quality*, so (a) the score distribution spreads (fixing the flat `score_std`), and (b) a decode threshold becomes meaningful and stable. Use Quality Focal Loss (QFL) form. *Expected:* `lane_score_std` rises well past 0.09; precision and threshold-ability improve; the F1 plateau lifts. *Risk:* low-medium; needs the per-match IoU available at loss time (it already is, via `line_iou`). *Grounding:* CLRerNet (WACV'24); QFL (Generalized Focal Loss, NeurIPS'20).

### S1.4 — Y-reweighted / scale-normalized regression (R1)
The per-row L1/LineIoU currently weighs all rows equally. Near-field rows have larger pixel error per unit angle. Either (a) weight the regression loss by row (heavier near camera) or (b) normalize each row's error by the local `dx/dy` so a fixed *angular* error costs the same everywhere. *Expected:* the model stops "giving up" on the near field (should also halt the `length_mean` shrink). *Risk:* low.

### S1.5 — Start-point spread fix (R1/R2, optional)
`startx_std ≈ 0.04` says priors collapse to center. Audit P3's square-prior re-init: confirm the 192 priors actually span left/center/right starts, and consider an **ADNet-style learnable/flexible start-point** so the model isn't forced to anchor every lane at center-bottom. *Expected:* off-center lanes improve, center over-commitment drops.

**Sprint 1 ablation grid (10k, ~15-ep probes):**
| run | S1.1 | S1.2 | S1.3 | S1.4 | watch |
|---|---|---|---|---|---|
| baseline (NB101 recipe) | – | – | – | – | F1, curveIoU, score_std, pred-count |
| +threshold | ✓ | – | – | – | precision, pred-count vs GT |
| +smooth | ✓ | ✓ | – | – | near-field jaggedness score |
| +iou-cls | ✓ | – | ✓ | – | score_std, threshold-ability |
| +yreweight | ✓ | – | – | ✓ | length_mean, near-field x-error |
| ALL | ✓ | ✓ | ✓ | ✓ | everything |

**Decision gate G1:** if "ALL" closes most of the gap (target: F1 ≥ ~0.32, near-field jaggedness halved, pred-count tracks GT) → promote to 70k, polish, and treat Sprint 2/3 as optional upside. If F1 is still ceilinged below ~0.30 despite the cls now being threshold-able → the *representation* is the binding constraint; go to Sprint 2.

---

## 6. Sprint 2 — Representation upgrades (≈2-3 weeks; pick by G1)

### S2.A — CLRerNet recipe (lowest risk, same family) ⭐ recommended first
Port **LaneIoU** (angle-aware IoU) into both the matcher (`assign`) and the regression loss (replace/augment `liou_loss`), and adopt the confidence calibration from S1.3 fully. CLRerNet reports clear gains over CLRNet on CULane and especially **CurveLanes** (the curvy split, which is exactly BDD's hard case). *Why it should work here:* your LineIoU uses a fixed thickness band that is blind to local angle, so it under-penalizes near-field shape error — LaneIoU fixes precisely that. *Effort:* days, not weeks — it's a loss/matcher swap. *Grounding:* CLRerNet WACV'24.

### S2.B — Re-scored Bezier head, properly trained (R1 step-change)
If S0.1 shows the Bezier geometry was good but mis-measured: fix the metric, then train the cubic-Bezier head with the Sprint-1 cls fixes (threshold + IoU-aware target) layered on. A cubic Bezier is **C² smooth by construction — it physically cannot be jagged**, so it eliminates the sawtooth outright; the only question is whether it sacrifices F1. Add the **feature-flip fusion** trick from BezierLaneNet to stabilize the *center* lane specifically (the worst case you observed). *Grounding:* BezierLaneNet CVPR'22. *Note:* you already have the Bezier + LCM scaffolding (`extensions/bezier_lcm`), so this is mostly de-risking + re-measuring, not greenfield.

**Decision gate G2:** compare S2.A vs S2.B on 70k by curveIoU **and** a qualitative smoothness panel (NB103-style). Keep whichever wins on the joint (F1, smoothness, cardinality) objective. If both plateau below target → the anchor/prior paradigm itself is limiting; go to Sprint 3.

---

## 7. Sprint 3 — Bold bets (≈3-5 weeks; choose ONE)

These are paradigm changes that attack all three symptoms at once. Do at most one, after Sprint 1/2 have de-risked the cheap wins.

### S3.A — Curve-query DETR head, unified with the RT-DETR detection decoder ⭐ the elegant bet
Replace the 192 fixed anchor priors with **N learnable lane queries** decoded by a transformer, each predicting (existence, curve params = Bezier or polynomial), trained with **one-to-one Hungarian matching + an auxiliary one-to-many branch** (H-DETR / Group-DETR / DN-DETR style) for fast cls convergence. Cardinality becomes *native*: the "no-object" class is first-class, so the model learns to output 0…K lanes — directly killing the "always 8" failure. **MapTR (in `external_repos/`) is the working template** for variable-cardinality vectorized geometry with permutation-invariant point matching; port its assigner + point loss.
*Why this is the most elegant option:* the detection branch is already RT-DETR (a query-based transformer decoder). Lane queries can ride the **same** decoder paradigm — one unified set-prediction head emitting boxes *and* lane curves, sharing the GCA-adapted features. This also resolves the multi-task story cleanly (both tasks become set prediction).
*Risk:* highest (new head, matching, longer convergence) — but the auxiliary one-to-many branch is the known cure for DETR's slow cls, and you've already validated Hungarian helps here.
*Grounding:* DETR; Deformable-DETR; DN-/Group-/H-DETR (hybrid matching); MapTR; Sparse Laneformer (a lighter lane-specific instantiation).

### S3.B — GANet keypoint + global association head (paradigm alternative)
Predict dense lane keypoints + each keypoint's offset to its lane's start; cluster keypoints by predicted start to form instances. Cardinality is emergent (no fixed count); the local per-keypoint regression is **naturally smooth** (no per-row independent x); excels on curved/dense scenes. *Risk:* medium-high (different output head + post-processing), but it sidesteps both the anchor-collapse and the per-row-wiggle problems. *Grounding:* GANet CVPR'22.

---

## 8. Sequencing & decision tree

```
Sprint 0 (instrument, 2d)
  ├─ S0.1 fix Bezier metric → re-score Bezier ckpts
  ├─ S0.2 score histogram (is cls threshold-able?)
  ├─ S0.3 threshold sweep on current model
  └─ S0.4 near/far jaggedness numbers
        │
        ▼
Sprint 1 (cheap, 1wk): threshold decode + smoothness reg + IoU-aware cls + y-reweight
        │
   G1: F1 ≥ ~0.32 AND jaggedness halved AND pred-count ~ GT ?
        │                                   │
       YES → promote 70k, polish.          NO → representation is binding
        │                                   │
        ▼                                   ▼
   (optional upside)                  Sprint 2: S2.A CLRerNet LaneIoU  (try first)
                                           and/or S2.B re-scored Bezier
        │
   G2: target met on (F1, smoothness, cardinality) ?
        │                                   │
       YES → ship.                         NO → Sprint 3 (pick ONE):
                                              S3.A curve-query DETR (unify w/ detection)
                                              S3.B GANet keypoints
```

**Recommended default path:** S0 → S1(ALL) → if needed S2.A (CLRerNet) → if needed S3.A (curve-query DETR). This front-loads the cheap, low-risk wins and only escalates to a new head if the data says the representation is the wall.

---

## 9. Target metrics (definition of done)

| metric | now (NB101) | Sprint 1 target | Sprint 2/3 target |
|---|---|---|---|
| lane curve-F1 @ IoU 0.5 | 0.236 | ≥ 0.32 | ≥ 0.45 |
| curveIoU (broad pool) | 0.091 | ≥ 0.15 | ≥ 0.25 |
| near-field jaggedness (mean sq 2nd-diff of x) | TBD (S0.4) | ≤ 50% of baseline | ≤ 25% |
| pred-lane-count vs GT-count corr. | ~0 (always 8) | ≥ 0.6 | ≥ 0.8 |
| precision / recall balance | 0.24 / 0.40 | within 1.3× of each other | balanced |
| detection mAP50 (guardrail) | ~0.50 | hold ≥ 0.48 | hold ≥ 0.48 |

Detection mAP50 is a **guardrail**: any lane change that drops it below ~0.48 reintroduces negative transfer and is rejected (this is exactly why `abl_asym_lr` was dropped).

---

## 10. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| Threshold helps precision but recall craters (cls too flat) | med | S1.3 IoU-aware cls is the real fix; threshold alone is a band-aid |
| Smoothness reg over-smooths genuine sharp curves | med | sweep weight; weight by row so far-field (already smooth) is untouched |
| Bezier re-score still low after metric fix | low-med | then Bezier truly underperforms; fall back to CLRerNet polyline (S2.A) |
| CLRerNet LaneIoU port has subtle frame/normalization bugs | med | unit-test LaneIoU against CLRerNet repo values on toy lanes before training |
| Curve-query DETR converges too slowly | med-high | auxiliary one-to-many matching (H-DETR) + DN queries; start on 10k overfit probe |
| Lane change drops mAP50 (negative transfer returns) | med | mAP50 guardrail in every ablation; keep aux_seg + gca_floor that already protect the trunk |
| Metric/decode drift between training-CSV and revalidation | high (already happened) | standardize ONE decoder used by training metric, validator, and NB103 |

> The last row is important: the Bezier-F1=0 artifact and the older runs' F1=0-in-CSV-but-0.211-after-NB100 both came from **multiple inconsistent decoders**. Before Sprint 1, unify on a single `_decode_row` shared by the loss-time metric, the validator, and NB103, so every number is comparable.

---

## 11. Why this is the right focus (framing)

NB94-101 systematically explored the **multi-task / negative-transfer** axis — matching schemes, GCA floors, aux segmentation, PCGrad, asymmetric LR, uncertainty weighting, self-distillation, full-data. The winner (aux_seg + gca_floor) holds detection mAP50 at ~0.50 and lifted lane-F1 to ~0.24. That axis is now near-exhausted: the trunk is healthy and shared-feature tricks give diminishing returns.

The plateau, the jaggedness, and the cardinality failure are **all lane-head-internal**: a coarse per-row polyline representation and an uncalibrated, threshold-free set-prediction decode. The plan therefore moves the effort from "share features better" to "**represent lanes better and predict the lane set properly**" — grounded in the lane-detection literature that solved these exact problems (CLRerNet for IoU/calibration, BezierLaneNet for smooth curves, MapTR/DETR for variable cardinality, GANet for keypoint smoothness).

---

## 12. Immediate next actions (for the following session, when code edits resume)

1. **Unify the decoder** (`_decode_row`) across metric/validator/NB103; fix the Bezier pred-gating bug.
2. **Sprint 0 notebooks**: re-score Bezier ckpts; score histograms; threshold sweep; near/far jaggedness.
3. **Clone references** into `external_repos/`: CLRerNet, BezierLaneNet (or pytorch-auto-drive), GANet, CondLaneNet (+ PDFs).
4. Draft the **Sprint 1 ablation grid** as a single parametrized notebook (threshold / smooth / iou-cls / y-reweight flags), 10k/15-ep probes.
5. Hold the **mAP50 ≥ 0.48 guardrail** in every run.

*(All of the above are proposed steps; this document changes no code.)*
