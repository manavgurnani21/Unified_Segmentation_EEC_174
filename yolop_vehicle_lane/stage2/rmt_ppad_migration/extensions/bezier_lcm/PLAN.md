# Bezier + LCM extension - implementation plan

> **NOTE (2026-05-30):** This is the original *pre-implementation design
> doc*. The brief ablation has since run (NB92) and revealed a dominant
> bug (the "5% stub" t-range init) that explains why bezier underperformed
> and why LCM hurt. For the actual results, root cause, the fix, and the
> updated experiment plan, see
> **`BEZIER_LCM_POSTMORTEM_AND_PLAN.md`** — it supersedes the results
> sections below. The design rationale here is still accurate history.

This document plans the next two notebooks (NB89 brief ablation, NB90
full training) based on:

- `tutorial/appendix-path3-bezier-topology-extension.md` (replace 78-D
  polyline with 16-D cubic Bezier)
- `tutorial/appendix-path3-lcm-experimental.md` (add GCA-style Lane
  Complexity Module choosing Bezier degree per lane)

It also addresses the GPU memory underutilization observed in NB88's
current run (13.3 / 95.6 GB on an RTX Pro 6000 at batch=8).

---

## 1. What the appendices propose (1-page summary)

### 1.1 Bezier topology extension

Replaces CLRKDNet's 78-D per-lane vector `[2 cls + 4 geom + 72 x-offsets]`
with a 16-D vector:

```
[ 0..1]   neg/pos scores
[ 2..5]   P0.x, P1.x, P2.x, P3.x   (cubic Bezier control points)
[ 6..9]   P0.y, P1.y, P2.y, P3.y
[10..11]  t_start, t_end           (validity range in [0, 1])
[12]      curve complexity score   (heuristic / monitoring)
[13..15]  reserved (LCM uses these)
```

Per appendix sec 0:
- 5x less per-prediction memory (78 -> 16 floats)
- C^infinity smooth by construction
- Renders at arbitrary density (model decides complexity via curvature)
- Naturally handles BDD-style sharp turns and partial visibility

### 1.2 LCM extension (built on top of Bezier)

A small GCA-style module (~30-50K params) that, per prior, outputs a
softmax over Bezier degrees `{K=1 straight line, K=2 quadratic, K=3
cubic}`. The final curve at training time is a soft mixture using the
SAME 4 control points interpreted at three degrees. An Occam's-razor
complexity penalty (gamma ~0.01) nudges the model toward simpler curves
where the IoU allows.

Mirrors GCA: small dedicated module, soft gating, monitoring scalar
(`complexity_score = mean effective K`), clamps to prevent degenerate
collapse.

### 1.3 Why these matter on BDD100K (vs CULane where CLRKDNet was tuned)

- BDD has sharp turns, S-curves, partial visibility - polyline assumes
  `x = f(y)` is monotonic in y, which breaks in turns
- Uniform 72-y sampling wastes capacity on straight segments and under-
  samples curves
- Per-lane "right complexity" varies by an order of magnitude; LCM lets
  the model adapt instead of fixing K=3 globally

---

## 2. Why this needs an implementation phase BEFORE NB89

The Bezier extension alone is described as "~+1 week on top of Path-3"
in appendix sec 11. LCM adds "~1-2 days" on top. Concretely, we have to
write new code that doesn't exist anywhere in the repo today:

| Module | New file | Replaces / Adds |
|---|---|---|
| Bezier least-squares fit | `B1_data/fit_bezier.py` | Replaces `vertices_to_clr_vector.build_full_target_tensor_v2` |
| Bezier rasterizer | `B7_eval/bezier_rasterize.py` | New (eval-time mask gen) |
| Bezier mixture renderer | `Bops/bezier_ops.py` | New (`render_bezier_mixture` and friends) |
| Bezier-aware loss | `B5_loss/bezier_losses.py` | Replaces lane_losses.py's xytl + LineIoU |
| Bezier-aware head | `B4_head/lane_bezier_head.py` | Replaces `LaneSegHead` (12-dim reg_layers, prior-to-bezier init, pool along Bezier curve) |
| LCM module | `BLcm/lane_complexity_module.py` | New (small GCA-style) |
| Bezier dataset adapter | `B6_data/dataset.py` patch | `update_labels_info` returns (B, max_lanes, 16) instead of 78 |

The Path-3 polyline pipeline stays intact in `P1-P8`; the Bezier track
is a parallel implementation under `extensions/bezier_lcm/`. NB88's
polyline run can continue without disturbance; we wait for its log,
then start writing the Bezier code, then run NB89.

---

## 3. Brief ablation - NB89 design

### 3.1 Ablation matrix (intentionally TIGHT)

Per the appendices, there are ~15 possible ablation rows across the two
tracks. For a BRIEF experiment that picks a winner before NB90, we
collapse to the three rows that answer the actual question - "does
Bezier help over polyline, and does LCM help over plain Bezier":

| # | Name | Representation | LCM | Loss | Rendering | Hypothesis |
|---|------|----------------|-----|------|-----------|------------|
| 1 | `polyline_baseline_small` | 78-D polyline (CLRKDNet original) | n/a | xytl + LineIoU | fixed 72 y | What NB88 row 2 measures, but on the same subset/budget as rows 2-3 below for an apples-to-apples comparison |
| 2 | `bezier_cubic_no_lcm` | 16-D cubic Bezier | OFF (always K=3) | bezier_geom + bezier_iou | fixed 72 uniform t | Does the representation itself help? |
| 3 | `bezier_lcm_gamma001` | 16-D cubic Bezier | ON, gamma=0.01 | + complexity penalty | mixture render | Does letting the model pick K=1/2/3 help on top? |

Skipped (per appendix sec 5.9 / 6.6 recommendations as "deferred until
the core wins are confirmed"):
- Bezier-K=6 quintic (diminishing returns)
- LCM density factor (orthogonal extension)
- LCM Gumbel hard selection (only if Strategy A is too slow at infer)
- LCM K-supervision from P1 (only if LCM doesn't converge in 50 ep)
- LCM at all stages vs final-only (we use final-only default)
- Smoothness loss (only if curves come out unstable)
- Validity range vs fixed [0, 1] (orthogonal; defer to NB90 winner)

### 3.2 Budget per row

The full P8 250-epoch run on the full 70k train set is taking 8 h /
19 epochs = ~25 min/epoch at batch=8 on an RTX Pro 6000. Across 3 rows
at 250 epochs each, that is ~150 GPU-hours - way too long for a "brief"
ablation.

**Brief budget**: 30 epochs each on a 10k-image stratified subset of
BDD train (using the same `prepare_bdd_subset.py` pattern from NB88,
just with a smaller image list). With the GPU memory plan in sec 4
below, batch=32 fits comfortably. Estimated wall time:

```
10000 train / 32 batch = ~310 batches/epoch
~310 * 6.5 it/s = ~50 s/epoch   (the throughput we saw in NB88 was
                                  ~4.5 it/s at batch=8; bigger batch
                                  reduces per-iter overhead modestly)
30 epochs ~= 25 min per row
3 rows = ~75 min training + ~30 min eval + ~30 min overhead
       = ~2.5 h total
```

That fits in one Colab session, leaves room for one re-run if anything
trips.

### 3.3 Why a 10k subset is informative enough

The point of NB89 is to RANK the three rows, not to produce final
numbers. Two consequences:

1. The ranking from 30 epochs on 10k images is a reliable predictor of
   the ranking from 250 epochs on 70k (this is standard ablation
   practice; see CLRKDNet's own paper sec 4.3 which uses 30-epoch
   probes on 1/7 the data).
2. Final numbers come from NB90 on the full dataset for the winner only.

If two rows tie within 0.5 IoU points at 30 epochs we extend the brief
ablation to 50 epochs and re-rank. If they still tie, take both into
NB90 (run 2 full trainings) and let the full-data numbers decide.

### 3.4 Cell-by-cell sketch (NB89)

```
Cell 1   Mount + locate sources                    (self-contained)
Cell 2   Build /content/bdd_10k_subset/            (new: 10k stratified sample)
Cell 3   Convert subset to BEZIER targets          (new: fit_bezier.py)
         + sanity-check Bezier fit error histogram
Cell 4   Helper launch_ablation_brief(...)         (parallel to NB88's)
Cell 5   Row 1: polyline_baseline_small            (reuses existing CLR pipeline)
Cell 6   Row 2: bezier_cubic_no_lcm                (new bezier head, lcm disabled)
Cell 7   Row 3: bezier_lcm_gamma001                (lcm enabled)
Cell 8   Aggregate: read each row's results.csv +
         best.pt; print per-row IoU/F1; announce winner
```

### 3.5 Acceptance criteria

NB89 PASSES if:
- All 3 rows reach epoch 30 without LossMonitor stops
- Per-row results.csv has a non-trivial IoU curve (not flat-zero, not
  flat-one)
- Bezier fit error (sec 3.4 cell 3) median < 3 px, p95 < 8 px
- Winner has at least +1 IoU point over second place, OR ties within 1
  point (triggering the "extend to 50 epochs" rule above)
- For row 3 specifically: `complexity_score` is meaningfully different
  from 3.0 (i.e., LCM is being used, not stuck at always-K=3)

---

## 4. GPU memory plan (applies to NB89 + NB90 + every future run)

### 4.1 Current state

NB88 row 2 (`clr_lane_default`): 13.3 / 95.6 GB at batch=8 = **14%
utilization**. On the RTX Pro 6000 (95.6 GB HBM3), this means we are
leaving roughly 7-8x the per-step batch on the table.

Cost: the 250-epoch run is projected at 105 hours wall time. At
batch=32 we expect ~30 hours; at batch=48 maybe ~22 hours. Per-image
gradient noise gets lower with larger batch, so the LR has to scale up
to maintain the same effective update; the standard recipe is "linear
LR scaling".

### 4.2 Plan for NB89 + NB90

| Setting | NB88 (current) | NB89 (brief) | NB90 (full) |
|---|---|---|---|
| `batch` | 8 | **32** | **32** (or **48** if NB89 shows headroom) |
| `lr0` | 1e-4 | **4e-4** (4x linear scale) | **4e-4** (or 6e-4 at batch=48) |
| `workers` | 4 | 8 | 8 |
| `imgsz` | 640 | 640 | 640 |
| Expected GPU mem | 13 GB | ~45 GB | ~45-65 GB |

Notes:
- AMP stays OFF (the NaN-at-warmup investigation in iter-0 of
  `debug_record.md` showed FP16 can't hold first-epoch DETR loss
  spikes; we keep FP32 even at the larger batch).
- `warmup_epochs=5.0` and `warmup_bias_lr=0.0` from NB88 carry over.
- LR scaling rule: linear up to ~256 effective batch; sqrt scaling
  thereafter. Batch 32 is in the linear region so 4x is fair.
- One concern: BatchNorm-style statistics in GCA's `task_adapter` and
  the CLR head's adapter benefit modestly from larger batch (more stable
  mean/var estimates). Should be a free win.

### 4.3 What to check on the first epoch of NB89

After cell 5 (row 1) starts, watch the run's `nvidia-smi` once:
- If mem < 50 GB and throughput is good: we can push batch up further
  in cells 6, 7 to save time
- If mem > 80 GB: we're close to the limit; stay at 32
- If LossMonitor fires on row 1 with the new LR, scale lr0 down to 2e-4
  (sqrt scaling instead of linear) and re-run

---

## 5. Full-training - NB90 design

NB90 takes the winner from NB89, and:

1. Runs the SAME config on the FULL 70k train + 10k val subset
   (extracted via the existing `prepare_bdd_subset.py` "full" mode).
2. Trains 250 epochs with the same hyperparameter envelope.
3. Produces the final result row that goes into the paper / report.

Cell sketch:
```
Cell 1   Mount + locate sources                    (self-contained)
Cell 2   Extract full BDD to /content/             (existing pattern)
Cell 3   Convert full set to BEZIER targets        (reuses NB89 Cell 3)
Cell 4   Single launch: winner config, 250 epochs, batch=32, lr0=4e-4
Cell 5   Aggregate: results.csv -> final numbers
```

Wall-time projection: ~30 hours at batch=32, single run. If the user
prefers, NB90 can also re-run the polyline baseline at batch=32 for a
clean apples-to-apples final number to put in the report next to the
Bezier winner.

---

## 6. Implementation phases (work that must happen between NB88 finishing and NB89 starting)

These are the new pieces of code that need to exist. They live under
`extensions/bezier_lcm/` to keep them separate from the P0-P8 polyline
pipeline (which stays a frozen "control" implementation).

| Phase | Output | Approx LoC | Effort |
|---|---|---|---|
| **B1** | `fit_bezier.py` - least-squares cubic fit, validity range, target tensor builder, fit-error verifier | ~250 | 0.5 day |
| **B4** | `lane_bezier_head.py` - replaces LaneSegHead. 12-dim `reg_layers`, `priors_to_bezier_init`, `pool_prior_features` along Bezier curve | ~400 | 1 day |
| **B5** | `bezier_losses.py` - `lane_bezier_loss`, `bezier_iou_loss`, `bezier_distance_cost` for assign | ~200 | 0.5 day |
| **B6** | dataset patch for 16-D targets | ~50 | 0.25 day |
| **B7** | `bezier_rasterize.py` - eval-time mask gen (uniform + adaptive modes) | ~150 | 0.5 day |
| **BLcm** | `lane_complexity_module.py` + integration into B4's head + complexity penalty in B5's loss | ~250 | 1 day |
| **Bops** | `bezier_ops.py` - `render_bezier_mixture`, `render_uniform`, `render_adaptive` | ~150 | 0.5 day |
| **Bsmoke** | scripts to smoke-test each phase individually (B1 fit on 10 images, B4 forward, B5 loss on random tensors, BLcm forward) | ~200 | 0.25 day |
| **NB89** | the brief-ablation notebook itself | ~150 (cells) | 0.5 day |
| **NB90** | the full-training notebook | ~80 (cells) | 0.25 day |

**Total: ~5 days of implementation work + smoke tests**, before NB89
can run end-to-end. The work is partitionable: B1 + Bops first (pure
data-side), then B4 + B5 (model + loss together so they can be smoke-
tested as a pair), then B7 (independent), then BLcm (on top of B4+B5),
then NB89 wiring.

---

## 7. Risk register (specific to Bezier + LCM)

From appendix Bezier sec 7 and LCM sec 7, plus what we learned in P8:

| Risk | Mitigation |
|---|---|
| Bezier fit error > 10 px on real BDD lanes | Fall back to splitting at inflection (S-curve handling); track p95 in NB89 cell 3 |
| Model fails to bend P1, P2 (control points stuck near midpoint -> always straight line) | Increase `lane_bezier_geom_loss_weight`; add `lane_curvature_reward` term (opposite sign smoothness loss) |
| LCM degenerates to w_3 = 1.0 always | Increase gamma to 0.05; if still stuck, add K-supervision from P1 |
| LCM oversimplifies to w_1 = 1.0 always | Reduce gamma to 0.001; check if cls loss is dominating |
| First-epoch NaN at batch=32, lr=4e-4 | Same envelope as P8: amp=False, warmup_epochs=5, warmup_bias_lr=0; if still NaN drop to sqrt-scaled lr=2.8e-4 |
| GPU OOM at batch=32 (unlikely but possible) | Step down to batch=24, lr=3e-4 |
| Brief-ablation result inconclusive (all 3 rows within 0.5 IoU) | Extend NB89 to 50 epochs OR put both top rows into NB90 |

---

## 8. Decision points along the way

1. **When NB88 finishes**: if `clr_lane_default` (polyline) reaches an IoU
   that already beats the BDD lane SOTA on the validation set, the
   business case for moving to Bezier weakens. In that case the user
   may want to invest the implementation effort in OTHER ablation rows
   (e.g., the deferred `clr_lane_attn_h2`) instead. Decision held until
   the log lands.
2. **After B1 verifier**: if median Bezier fit error > 5 px on 100
   random BDD lanes, stop and revisit P1's polyline extraction (the
   skeleton may be too noisy for a 4-control-point fit).
3. **After NB89 cell 8**: pick winner and write `extensions/bezier_lcm/
   ablation_brief_results.md` BEFORE launching NB90. The user will
   review and confirm before the 30-hour full run starts.

---

## 9. What is NOT changing

- NB88 stays untouched. The polyline P0-P8 pipeline stays as the
  control / reference implementation. If Bezier ends up not beating
  polyline at the full scale, we publish both numbers and keep the
  polyline as the production model.
- The P5 loss weight rescale (`* 0.2`) and `diff.clamp(+/-100)` fix from
  `debug_record.md` carry over to the Bezier-pipeline loss as the same
  conservative defaults.
- The LossMonitor with the EMA cap stays - in fact it's MORE useful for
  Bezier since the first-epoch curve-fitting can be even more spiky than
  the polyline case.
