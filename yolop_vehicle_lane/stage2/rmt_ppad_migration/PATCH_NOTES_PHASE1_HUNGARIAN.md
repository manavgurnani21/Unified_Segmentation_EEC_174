# Phase 1 patch notes - Hungarian matching + lane-cls liveness diagnostics

Implements the highest-leverage fix from PLAN_AFTER_NB88_89_92.md plus the
three diagnostics the user requested, so we can both fix the frozen-IoU
collapse and automatically detect if it ever recurs.

## 1. Hungarian 1-to-1 matching (the fix)

Root cause recap: dynamic-k matching labels a prior positive in some
batches / negative in others -> the binary cls converges to a uniform
0.5 sigmoid -> every prior clears any threshold equally -> decoded lane
IoU freezes (0.0718 forever in NB88/89/92). Hungarian gives each GT
exactly one prior, deterministically, so the cls gets a stable positive
label and can separate (proven in Exp2RR/NB47: pos-neg gap 0.01 -> 0.099).

- `P5_loss/tools/lane_losses.py`: new `hungarian_assign(cost)` (scipy
  `linear_sum_assignment`); `assign(..., match='hungarian')` (default).
- `extensions/bezier_lcm/B5_loss/tools/bezier_losses.py`: new
  `hungarian_assign_bezier`; `assign_bezier(..., match='hungarian')`.
- `vendor/.../models/utils/loss.py`: `MTDETRDLoss` reads `LANE_MATCH`
  env var (default `hungarian`), passes `match=self.lane_match` to both
  assign call sites. Env var avoids plumbing a kwarg through Ultralytics'
  deep call stack.
- `P8_train/scripts/train_lane_only.py` and
  `extensions/bezier_lcm/scripts/train_bezier_brief.py`: new
  `--lane-match {hungarian,dynamic_k}` flag (default hungarian) that sets
  `os.environ['LANE_MATCH']` before importing ultralytics.

Revert path: `--lane-match dynamic_k` restores the original scheme.

## 2. Top-N rasterization for IoU (remove the threshold)

`P7_validator/tools/lane_rasterize.py`: new `_select_topn(...)` helper.
When `conf_threshold` is None or <= 0 the absolute cutoff is removed and
we keep purely the top-`max_lanes` priors by score RANKING. Both
`lanes_to_mask` (78-D) and `_bezier_lanes_to_mask` (16-D) route through
it. `vendor/.../models/mtdetr/val.py` now calls with
`conf_threshold=None`.

Why: with cls collapsed to ~0.5, every prior clears any cutoff equally
and the top-k tie breaks identically each epoch -> frozen IoU. Pure
top-N makes the decoded mask depend ONLY on the ranking, so the instant
the cls starts to separate (even within a tight 0.5 cluster) the
selected set shifts and IoU responds. IoU becomes a sensitive "is the
cls moving?" instrument.

## 3. Per-prediction score histogram (visual dead-head check)

- `P7_validator/tools/lane_rasterize.py`: new `lane_score_stats(preds)`
  returns min/max/mean/std/spread + a coarse histogram of the 192 priors'
  pos-scores.
- `vendor/.../models/mtdetr/val.py`: accumulates pos-scores across the
  val pass (`_accum_lane_scores`) and, once per epoch in `get_stats`,
  prints:
  ```
  [lane-score] n=384000 min=0.31 max=0.74 mean=0.49 std=0.04210 spread=0.43
  [lane-score] hist[0..1] |  .:=*#%@*=:.  |  (peak=...)
  ```
  Eyeball whether the 192 lines are still clustered at 0.5 (dead) or
  spreading (learning).

## 4. Automatic dead-lane monitor

- `vendor/.../models/mtdetr/val.py`: surfaces
  `metrics/lane_score_std(lane)`, `metrics/lane_score_spread(lane)`,
  `metrics/lane_score_mean(lane)` into the per-epoch metrics dict (so
  they land in results.csv too), and prints an immediate WARNING when
  per-pass std < 1e-3.
- `training_utils/tools/epoch_callbacks.py`: the per-epoch metrics table
  gains `scoreStd` / `scoreSprd` columns, and a `[dead-lane-monitor]`
  fires if `lane_score_std` stays < 1e-3 for 5 consecutive val epochs
  (plus a secondary note if IoU is byte-frozen).

## 5. Probe notebook

`notebooks/stage2_notebook_94_hungarian_probe.ipynb` (+ builder
`nb94_build_helper.py`): runs the polyline head on the 10k subset twice -
`--lane-match hungarian` vs `dynamic_k` - and Cell 6 prints both IoU +
scoreStd trajectories side by side with a PASS/INCONCLUSIVE verdict.

## Pass criteria (run NB94 to evaluate)

- Hungarian row: IoU(lane) climbs past 0.10 and rises across 30 epochs;
  scoreStd grows visibly away from 0.
- dynamic_k row: IoU frozen ~0.07, scoreStd ~0, `[dead-lane-monitor]`
  fires.

If Hungarian-on-anchor is insufficient (both rows frozen), escalate to
the K=64 query head (Exp2RR) per PLAN_AFTER_NB88_89_92.md Phase 1
fallback.

## Files touched
- P5_loss/tools/lane_losses.py
- extensions/bezier_lcm/B5_loss/tools/bezier_losses.py
- vendor/RMT-PPAD/ultralytics/models/utils/loss.py
- vendor/RMT-PPAD/ultralytics/models/mtdetr/val.py
- P7_validator/tools/lane_rasterize.py
- vendor/RMT-PPAD/ultralytics/models/utils/lane_rasterize.py  (shim re-export)
- training_utils/tools/epoch_callbacks.py
- P8_train/scripts/train_lane_only.py
- extensions/bezier_lcm/scripts/train_bezier_brief.py
- notebooks/nb94_build_helper.py + stage2_notebook_94_hungarian_probe.ipynb (new)
