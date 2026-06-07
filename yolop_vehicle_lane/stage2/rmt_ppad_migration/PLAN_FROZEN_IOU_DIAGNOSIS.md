# Frozen lane-IoU: problem definition + step-by-step diagnostic plan

Date 2026-05-30. After NB94 (Hungarian probe). Numbering rule (user):
every DIAGNOSTIC / ABLATION notebook below is numbered BEFORE any
notebook that does a full-epoch training run. Diagnostics = NB95-98
(all short: <=3 epochs, <=2k images). Full training = NB99+ only AFTER
the root cause is pinned.

---

## 1. The decisive evidence (NB94)

NB94 ran the polyline head with **Hungarian matching confirmed active**
(`[MTDETRDLoss] lane matching scheme = hungarian`, `LANE_MATCH=hungarian`,
scipy present). Result:

```
ep  IoU(lane)  scoreStd  scoreSprd
 1   0.0710    0.0075    0.0815
 2   0.0710    0.0616    0.4217
 3   0.0710    0.0708    0.3301
 4   0.0710    0.0724    0.4006
[frozen-iou-monitor] *** DECODED IoU(lane) FROZEN *** at 0.0710 for 3
  consecutive val epochs while cls score_std=0.07236
```

- **cls IS learning**: pos-score std rises 0.007 -> 0.072, spread widens.
- **decoded IoU is byte-frozen** at 0.0710 (subAcc frozen at 0.5000).

**Conclusion: the matching scheme was NEVER the cause.** Hungarian fixed
the cls (as the prior research predicted) but IoU is still frozen. The
freeze is downstream of cls, in the GEOMETRY -> EVAL -> RASTERIZE -> IoU
path.

The frozen value is a per-val-set constant: 0.0710 (10k subset) vs
0.0718 (70k). A constant that depends only on the val set, not the
epoch, is the signature of a **static predicted mask** being compared to
a fixed GT.

---

## 2. What the reference repos do (cross-repo audit)

### CLRKDNet (the head we ported)
- Eval returns `predictions_lists[-1]` (raw 78-D tensor), then
  **`get_lanes()` decodes**: softmax conf-filter (>=0.4) -> **custom NMS
  (top_k=max_lanes)** -> `predictions_to_pred()` -> Lane objects via
  **InterpolatedUnivariateSpline**.
- Its lane METRIC is **curve-based F1**: decode -> spline-sample at fixed
  ys -> rasterize each curve with `cv2.line(width=30)` -> pairwise pixel
  IoU -> Hungarian match pred<->GT -> TP/FP/FN at IoU thresholds -> F1@0.5.
- Geometry DOES learn there because the whole pipeline (NMS+spline) feeds
  a real curve metric; the loss uses line_iou + smooth_l1 with weights
  cls=2.0, xyt=0.2, iou=3.0.

### RMT-PPAD (the multi-task host)
- Its seg branch is a **DENSE per-pixel decoder**
  (TransformerSegmentationDecoder): `mask = sigmoid(logits) > 0.5`. Every
  pixel is an independent logit that learns directly. IoU =
  `SegmentationMetric.IntersectionOverUnion()[1]` = TP/(TP+FP+FN) for the
  lane class on the dense mask.
- "Frozen/constant IoU == the predicted mask is static across batches."

### Our port = a HYBRID of the two, and that is the problem
We took CLRKDNet's **sparse anchor head** and fed its output into
RMT-PPAD's **dense pixel-IoU metric** via our own `lanes_to_mask`:
- We rasterize the **top-N RAW anchor predictions** (NO NMS, NO spline
  decode) into a 640x640 mask.
- We compute pixel IoU of that mask vs the rasterized GT lane mask.

This hybrid breaks in two compounding ways (below).

---

## 3. Problem definition (two compounding faults)

### Fault A - the GEOMETRY (reg_layers) is not learning
cls gets a DENSE signal (FocalLoss over all 192 priors, matched=1 /
unmatched=0). reg gets a SPARSE + WEAK + possibly-CLAMPED signal:
- only matched priors are supervised (Hungarian = ~1 prior/GT = ~4/image)
- weights were cut 5x (xytl 0.5->0.1, iou 2.0->0.4)
- the xytl `diff.clamp(-100,100)` zeroes the gradient for any matched
  prior whose start_x is >100 px from its target (likely early on)

Net: reg_layers barely move, so the eval predictions stay ~= the prior
anchors, so the rasterized eval mask = the fixed anchor fan = constant
mask = frozen IoU. (NB95 will MEASURE this directly via the new
`[lane-geom]` log I just added to val.py.)

### Fault B - the metric itself is a poor/saturated proxy
Even if reg learned, our metric rasterizes the **top-N RAW predictions
without NMS/spline**, so:
- the top-N by cls score need not be the geometrically-good priors
- no NMS means near-duplicate priors crowd the top-N
- pixel IoU of thin polylines vs thin GT lines is dominated by near-misses
This is NOT how either reference measures lanes. CLRKDNet's curve-F1 is
the correct yardstick; our pixel-IoU may stay near-constant even while
the model improves.

**The frozen 0.0710 is most likely Fault A (static anchor mask), but
Fault B must be ruled out because it would make even a fixed model look
"stuck" and would mask real progress.**

---

## 4. Step-by-step diagnostic plan (NB95-98, all SHORT)

Each notebook is small (<=3 epochs, 2k-image subset) and writes its
verdict to `extensions/diagnostics/<nb>_result.md`. Run them in order;
each gates the next. NO full-epoch training until NB98 concludes.

### NB95 - "Is the geometry frozen?" (instrumentation read-out)
- Run the polyline head 3 epochs on 2k images with the new val.py
  `[lane-geom]` logging (start_x/theta/length mean+std per epoch).
- PASS criterion to ADVANCE: confirm whether `lane_startx_mean`,
  `lane_theta_mean`, `lane_length_mean` are byte-identical across epochs.
  - If CONSTANT -> Fault A confirmed (geometry frozen). Go to NB96.
  - If MOVING but IoU still frozen -> Fault A is not it; jump to NB97
    (metric fault).
- Also dump, for 1 batch, `max|eval_pred - prior_anchor|` over the
  geometry fields. ~0 => eval == anchors.

### NB96 - "Why is reg not learning?" (geometry-gradient ablation)
Only if NB95 shows frozen geometry. Four 2-epoch micro-runs, each
toggling ONE lever, comparing `lane_startx_std` movement + matched
`lane_iou_loss` slope:
1. diff-clamp OFF (remove the +-100 xytl clamp)
2. geometry weights restored to CLRKDNet (xytl 0.2, iou 3.0) - undo 5x cut
3. lane-only (freeze detection backbone grads) - remove joint conflict
4. denser matching (dynamic_k top-4) vs Hungarian - more positive priors
- Verdict: which lever(s) make start_x/theta std actually grow. That
  is the geometry fix to carry into the full run.

### NB97 - "Is the metric lying?" (metric-correctness check)
Independent of NB95/96. Implement CLRKDNet's REAL curve metric as an
ALTERNATE column alongside pixel-IoU:
- decode eval preds with `get_lanes()` (conf-filter + NMS) ->
  `predictions_to_pred()` -> spline -> rasterize `cv2.line(width=30)` ->
  curve-F1@0.5 (Hungarian pred<->GT).
- Run 3 epochs; log BOTH pixel-IoU(lane) and curve-F1(lane).
  - If curve-F1 MOVES while pixel-IoU stays frozen -> Fault B confirmed:
    our metric is broken, the model was learning all along. Switch the
    headline metric to curve-F1.
  - If BOTH frozen -> it really is Fault A; the model is not learning
    geometry and NB96's fix is required.

### NB98 - "Smallest working config" (overfit sanity)
The decisive sanity check: can the lane head overfit a TINY set?
- 1 epoch is irrelevant; instead train 50 epochs on **16 images** with
  the NB96-winning levers, lane-only, full geometry weights, no clamp.
- A correct head MUST drive curve-F1 high (>0.5) and move pixel-IoU on
  16 images. 
  - If it overfits 16 images -> the architecture/loss is sound; scale up.
  - If it CANNOT overfit 16 images -> there is a structural bug (eval/
    train mismatch, gradient not reaching reg_layers, target encoding
    error). Fix that before any scaled run.

---

## 5. Only after NB95-98: full training (NB99+)
Carry the confirmed fixes (geometry weights, clamp policy, matching,
metric) into a full run. NB99 = full polyline-head training with the
corrected geometry supervision + curve-F1 metric. Bezier/LCM ablations
follow on the now-working substrate.

---

## 6. Immediate updates already shipped this iteration
- `val.py`: added `[lane-geom]` per-epoch logging of start_x/theta/length
  mean+std + 4 new `metrics/lane_*` keys. This is NB95's instrument; it
  needs zero new training code, just a re-run.
- (carried) frozen-IoU monitor now fires at patience 3 with the
  "geometry not learning" message.

## 7. What NOT to do
- Do NOT start another full-epoch training to "see if it helps" - NB94
  already proved Hungarian alone doesn't move IoU. Diagnose first.
- Do NOT re-cut or re-tune lane weights blindly; NB96 measures which
  lever matters.

---

## 8. REVISION (after review) - pruned to 2 notebooks, NB95 + NB96

Numbering intent (clarified by user): the narrative is
`full-training (NB88-94) -> these experiments -> full-training (after fix)`.
So the diagnostics get the NEXT numbers after the initial full runs
(NB95, NB96), and the post-fix full training is NB97+. Diagnostic numbers
stay BELOW the post-fix full-training numbers.

Pruned the original 4 to 2 (removed redundancy; the user said drop
notebooks whose outputs we won't reuse):
- DROP standalone "is geometry frozen?" - folded into the new val.py
  `[lane-geom]` logging, which now prints in EVERY run. No notebook needed.
- DROP standalone curve-F1 notebook - DEFERRED (conditional). The
  byte-EXACT frozen IoU is the static-mask signature of Fault A, not the
  Fault-B metric noise (a broken metric would wobble, not freeze to 5
  decimals). Build curve-F1 ONLY if NB95 shows geometry MOVING while
  pixel-IoU stays frozen. Until then it has no value to exist.
- MERGE "overfit-16" + "geometry readout" + "dual-metric" into ONE
  decisive capability probe (NB95).

### NB95 - lane-head overfit capability probe (DECISIVE)
The single most informative test: can the lane head learn lanes AT ALL?
- Train the polyline head on a TINY fixed subset (32 images), ~120
  epochs, with the BEST-CASE geometry config: `--lane-weights clrkd`
  (full CLRKDNet weights), `--diff-clamp none`, `--lane-match hungarian`.
- Watch `[lane-geom]` (does start_x/theta/length move?), `IoU(lane)`,
  `scoreStd`.
- Verdict:
  - IoU climbs + geometry means move => architecture/loss are SOUND;
    frozen IoU at scale is a weight/clamp/optimization issue => NB96
    finds the minimal fix that survives at 2k images.
  - IoU still frozen on 32 images => STRUCTURAL bug (train/eval forward
    mismatch, target-encoding error, gradient not reaching reg_layers).
    NB95 then IS the minimal repro to single-step debug; do not scale up.

### NB96 - geometry-lever ablation @ 2k images (FINDS THE FIX)
Only the levers NB95 didn't already settle. 4 rows x 3 epochs x 2k img:
| row | lane-weights | diff-clamp | match | question |
|---|---|---|---|---|
| A | cut5x | 100 | hungarian | reproduce NB94 baseline (control) |
| B | clrkd | none | hungarian | does strong unclamped geometry move IoU? |
| C | clrkd | none | dynamic_k | does denser matching help geometry? |
| D | clrkd | none | hungarian + lane-only* | does removing the det conflict help? |
(*lane-only row deferred unless A-C are inconclusive; needs a freeze flag)
- Output: the smallest config whose `IoU(lane)` and `[lane-geom]` move,
  to carry into the NB97+ full run.

### Then: NB97+ = post-fix full training (NOT part of diagnostics).

All knobs are env-driven CLI flags now on `train_lane_only.py`:
`--lane-weights {cut5x,clrkd}`, `--diff-clamp {100|none}`,
`--lane-match {hungarian,dynamic_k}`. No code edits between rows.
