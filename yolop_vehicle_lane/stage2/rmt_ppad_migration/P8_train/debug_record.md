# P8 training debug record

Iterative log of every failure → fix → outcome cycle for the full P8
training runs (NB88). Each iteration appended in chronological order with
a fixed schema:

```
## Iteration N - <short symptom slug>

### Symptom
What the user observed. Direct quote or paraphrase of the failure.
Include the run that exhibited it (Cell 4 / Cell 5 / Cell 6) and the
step / epoch where it manifested.

### Numeric evidence
The smallest set of numbers from the training log that pin the failure.

### Diagnosis
Why this happens. Trace from numbers -> code path.

### Fix
What was changed. Files + intent (the actual diff is in git).

### Expected behavior after fix
What should be visible in the next NB88 run if the fix lands.

### Observed behavior after fix
Filled in by the NEXT iteration once the user re-runs NB88. If still
broken, this also lists the new symptom and feeds Iteration N+1.

### Verdict
PASS / PARTIAL / FAIL. PARTIAL = the original symptom is gone but a
different one surfaced.
```

The point of this file: each new debug iteration can read the prior
iterations and decide whether to push harder on the same lever, switch
levers, or revert and try a different approach. Without this we keep
re-deriving the same diagnosis from scratch every time.

---

## Iteration 0 - ll_seg EMA 16 -> 240 in epoch 1 (PRE-summary baseline)

### Symptom
First full launch of `clr_lane_default` stopped by LossMonitor during
warmup of epoch 1 with `ll_seg divergence: EMA 16.5 -> 240 over last
500 batches`.

### Numeric evidence
```
1/250  13.2G  Detection=62.43  da=0  ll_seg=70.87  lr=1.37e-6  grad_norm=756
[loss-monitor] ll_seg divergence: EMA 16.5 -> 240 over last 500 batches
```
Detection healthy, lr still in warmup, grad_norm finite -> not a runaway
optimizer step; the lane loss itself was the issue.

### Diagnosis
CLRKDNet's lane loss weights `(cls, xytl, iou, seg) = (2.0, 0.5, 2.0, 1.0)`
are tuned for a ~5 M-param lane-only network. Plugged verbatim into our
35 M-param RT-DETR + CLR joint setup they made the *summed* `ll_seg`
~5x what the detection branch saw, so the shared backbone gradient was
dominated by lane updates from step 1.

### Fix
1. `vendor/RMT-PPAD/ultralytics/models/utils/loss.py` MTDETRDLoss
   weights cut 5x:
   ```python
   self.lane_cls_weight  = 2.0 * 0.2   # 0.4
   self.lane_xytl_weight = 0.5 * 0.2   # 0.1
   self.lane_iou_weight  = 2.0 * 0.2   # 0.4
   self.lane_seg_weight  = 1.0 * 0.2   # 0.2
   ```
2. `P8_train/scripts/train_lane_only.py` LossMonitor loosened:
   `warmup_batches: 200 -> 1000`, `divergence_growth: 1.5 -> 3.0`.

### Expected behavior after fix
- `ll_seg` should start around ~3 (5x smaller than 16)
- monotone-ish climb tolerated up to ~3x EMA over a 500-batch window
- no LossMonitor stop in epoch 1

### Observed behavior after fix
See Iteration 1 - the headline `ll_seg` magnitude did drop (3.5 instead
of 16), but a NEW failure surfaced: isolated single-batch outlier spikes
of 6000+ poisoned the EMA and re-tripped the divergence check.

### Verdict
PARTIAL. Weight rescale was necessary and correct; LossMonitor
loosening was insufficient on its own.

---

## Iteration 1 - transient single-batch xytl spikes (current)

### Symptom
NB88 cells 4, 5, 6 ALL stopped during epoch 1 with the same pattern: a
healthy-looking running ll_seg average (3 ~ 10) is repeatedly poisoned
by occasional single-batch values of 100 ~ 6000+.

### Numeric evidence
Cell 4 (`clr_lane_default`) - stopped at step 3100 / 8750:
```
step  565  spike: current=6.33e+03 ema=3.52  (consec=1)
step 1164  spike: current=141     ema=3.35  (consec=1)
step 2802  spike: current=14      ema=2.29  (consec=1)
step 3039  spike: current=613     ema=2.19  (consec=1)
step 3100  ll_seg divergence: EMA 2.28 -> 10.3 over last 500 batches
```
Cell 5 (`clr_lane_no_square_priors`) - stopped at step 2000:
```
step 1939  spike: current=1.33e+03  ema=2.69
step 2000  ll_seg divergence: EMA 3 -> 9.79 over last 500 batches
```
Cell 6 (`clr_lane_no_gca`) - stopped at step 2000:
```
step 1939  spike: current=2e+03    ema=2.7
step 2000  ll_seg divergence: EMA 3.07 -> 13.4 over last 500 batches
```
Detection column was healthy in all three (67 -> 27 monotonically).
Running-average ll_seg in the per-row header stayed in [3, 10].

### Diagnosis
Per-iter dive into `_compute_lane_only_loss`:

```python
reg_yxtl = predictions[matched_row_inds, 2:6].clone()
reg_yxtl[:, 0] *= n_strips         # start_y: [0,1] -> [0,71]
reg_yxtl[:, 1] *= (img_w - 1)      # start_x: [0,1] -> [0,639]
reg_yxtl[:, 2] *= 180              # theta:   [0,1] -> [0,180]
reg_yxtl[:, 3] *= n_strips         # length:  [0,1] -> [0,71]
xytl_loss = F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean()
```

CLRHead's reference implementation assumes `predictions[:, :, 2:5]` lives
in `[0, 1]` (the prior init range). The only thing keeping it there is
`nn.init.normal_(reg_layers.weight, std=1e-3)` and small inputs - it
relies on lane-only training and a low-variance backbone.

In our joint setup:
- The GCA adapter feeds high-variance features into CLRHead
- Detection gradients also poke the shared backbone
- After a handful of optimizer steps `reg[:, :, :3]` can output values of
  magnitude 10+ for SOME priors on SOME batches

When a prior's start_x drifts to, say, 5 in normalized units:
`reg_yxtl[:, 1] = 5 * 639 = 3195`; target is in [0, 639], so the diff
explodes. `smooth_l1(3195 - 320) = 2874.5`. Averaged over 4 elements and
~4 matched priors, summed over 8 batch x 6 stages, this single image
can contribute 6000+ to `xytl_loss_avg`. Multiplied by lane_xytl_weight
(=0.1) it shows up in `ll_seg` as ~600. Across multiple bad priors per
batch we hit 6000.

These spikes are TRANSIENT (single batches, well separated). The model
is otherwise learning fine. The LossMonitor's `ema_decay=0.99` means one
6330 event pulls `ema = 0.99*3.5 + 0.01*6330 = 66.8`, then it takes
hundreds of batches to decay back. Successive spikes keep the EMA
elevated, triggering the divergence check.

### Fix
1. `vendor/RMT-PPAD/ultralytics/models/utils/loss.py`
   `_compute_lane_only_loss` - clamp the smooth_l1 input diff to a sane
   pixel-magnitude range so a single out-of-range prediction can't
   contribute thousands to the loss:

   ```python
   diff = reg_yxtl - target_yxtl
   diff = diff.clamp(min=-100.0, max=100.0)
   xytl_loss = F.smooth_l1_loss(
       diff, torch.zeros_like(diff), reduction='none',
   ).mean()
   ```

   Smooth-L1 on a clamped diff: gradient through the clamp is 1 for
   `|diff| <= 100` (the normal regime) and 0 outside. The 1% of bad
   priors lose gradient signal; the 99% normal priors are unaffected.
   `xytl_loss` per (image, stage) is now bounded by 99.5; total
   contribution to `ll_seg` is bounded by ~10.

2. `P8_train/scripts/training_monitors.py` `LossMonitor.update()` -
   cap the per-batch value at `10 * EMA` before blending into the EMA.
   A single 6000 batch is still counted as a "spike" event (via the
   already-existing `consecutive_spikes` counter) but it can't pull
   the EMA from 3 to 67. This is belt-and-suspenders for any other
   loss term we might add later.

### Expected behavior after fix
- `ll_seg` per-batch now bounded by ~25 in the worst case
- LossMonitor spike WARNINGS may still fire occasionally (informative
  only, no stop) for the first ~1000 batches
- Cells 4, 5, 6 should clear epoch 1 without a LossMonitor-induced stop
- `Detection` should remain in [25, 70] across epoch 1
- `grad_norm` should stay under ~500

### Observed behavior after fix
*(filled in by Iteration 2 once user re-runs NB88)*

### Verdict
*(pending re-run)*

---

## Iteration 2 - SYSTEMIC: lane IoU frozen across all runs (NB88/89/92)

### Symptom
After NB88 (49 ep), NB89 (48 ep), NB92 (3x 30 ep) completed, inspected
`training_runs/checkpoints/*/results.csv`. Lane IoU is FROZEN:
- polyline runs: IoU(lane) = 0.0718 / 0.0710 byte-constant for all
  epochs; subacc(lane) = exactly 0.5000.
- bezier runs: IoU(lane) noisy, collapses to ~0.003.
Meanwhile training ll_seg DOES decrease (1.305 -> 0.593 on NB88).
Detection mAP peaks ~ep15 then declines every run; best_fitness locks
early.

### Diagnosis
NOT a new bug - this is the KNOWN "lane cls collapse" documented across
~70 prior experiments (exp01-71 logs + worktree PATCH_NOTES_EXP2G..VV).
Root cause: dynamic-k matching labels each prior pos in some batches /
neg in others -> binary cls converges to uniform sigmoid (0.5) -> all
priors pass conf_threshold -> rasterizer draws same anchor fan every
epoch -> IoU frozen. ll_seg drops only because CLRHead's aux seg-decoder
convnet (separate per-pixel head) trains; the prior-curve regression
that feeds decoded IoU does not. Confirmed both migration loss files use
dynamic_k matching.

### Fix
NO point-fix. Re-scoped via PLAN_AFTER_NB88_89_92.md:
- Phase 1: replace dynamic-k with Hungarian 1-to-1 matching (the only
  proven cls-collapse fix, Exp2RR/NB47: gap 0.01->0.099, f1 0->0.25).
- Phase 4: only ablate bezier-vs-polyline AFTER lane IoU learns.
- Do NOT promote any NB92 row to full NB93 training.

### Verdict
NB88/89/92 = substrate-validation PASS (pipeline/infra works), science
FAIL (lane head inert). Brief ablation invalid. Pivot to Phase 1.

---

## Iteration 3 - final_eval plot_predictions crash + early-terminated rows

### Symptom (from NB92 cell OUTPUTS, not just logs)
Inspected the .ipynb cell outputs directly:
- Cell 5 / Row 1 (polyline): "30 epochs completed in 1.868 hours" THEN
  crashed in final_eval -> plot_predictions -> output_to_target:
  `TypeError: list indices must be integers or slices, not tuple`.
- Cell 6 / Row 2 (bezier_cubic) + Cell 7 / Row 3 (bezier_lcm): output
  truncated to last 5000 lines; last content = epoch 19/30 and 20/30
  respectively, healthy loss, NO traceback, no return_code line.

### Diagnosis
1. plot_predictions: this validator's postprocess returns a 2-tuple
   `(detection_outputs_list, lane_mask)`. plot_predictions passed the
   whole tuple to `output_to_target`, which iterates it and indexes the
   `outputs` LIST with `[:max_det, :6]` (a tuple index) -> TypeError.
   Only fires in final_eval (training-time val has plots off), so it
   waited until after a full 30-epoch run to bite. This is the SECOND
   final_eval landmine after the strip_optimizer lambda (Iteration ~).
2. Rows 2/3 stopping at epoch 19-20 with NO Python traceback = EXTERNAL
   termination (Colab session timeout / disconnect / manual stop), not a
   code bug. Three sequential 1.9h rows ~= 5.6h exceeds typical Colab
   limits. They had NOT reached final_eval yet, so this is unrelated to
   the plot crash - but they WOULD have hit it at epoch 30.

### Fix
`vendor/.../models/mtdetr/val.py` plot_predictions: extract `preds[0]`
when preds is a 2-tuple, AND wrap plot_images in try/except so a pure
visualization error can never again discard a completed training run.

### Process note (user feedback)
Inspect notebook .ipynb OUTPUTS, not only the *_train.log files - the
truncation + final traceback are visible in the cell outputs and reveal
both the crash and the early-termination.

### Verdict
final_eval path now clear: strip_optimizer (lambda->class) + plot_predictions
(preds[0]+try/except) both fixed. Per-epoch Drive sync already preserved
last.pt/best.pt/results.csv for every completed epoch, so no trained
weights were lost to either crash. Rows 2/3 just need a re-run (resume).

---

## Iteration 4 - NB92 cell 2 "exit status 2" = prep script missing on Drive

### Symptom
NB92 re-run: Cell 2 (extract 10k subset) failed with
`CalledProcessError ... returned non-zero exit status 2`. No other error
text - subprocess.check_call swallowed the child's stderr.

### Diagnosis
Reproduced locally: `python -u <missing_file.py>` prints
`can't open file ... [Errno 2] No such file or directory` to STDERR and
exits with code **2** (NOT an argparse error, NOT exit 1). check_call
hid that stderr, leaving only "exit 2". Root cause: the prep script
`extensions/bezier_lcm/scripts/prepare_bdd_subset_10k.py` (part of the
NEW Bezier code tree) is missing/stale on the user's Drive copy of
yolop_vehicle_lane. The whole extensions/bezier_lcm/ tree + recent
vendor edits need to be synced to Drive.

### Fix
NB92 cell 2 hardened (mirrors the NB88/89/90 cell-4 fix that NB92 never
got): (1) pre-flight `PREP_SCRIPT.exists()` -> clear FileNotFoundError;
(2) pre-flight the 3 Drive source files; (3) Popen with
stderr=subprocess.STDOUT streaming so the REAL error is always visible;
(4) RuntimeError with cause hints on non-zero exit. Regenerated via
nb92_build_helper.py.

### Verdict
Observability fixed + root cause identified (Drive sync gap). User must
upload the updated yolop_vehicle_lane (esp. extensions/bezier_lcm/ +
vendor loss.py/val.py/lane_losses.py) to Drive before re-running.

---

## Iteration 5 - IoU frozen WITH healthy cls; logs/warnings/monitor

### Symptom (from NB88/89 notebook OUTPUTS)
User: "IoU haven't changed", "training logs haven't updated", saw
UserWarnings, "monitor didn't work for unchanged IoU".

### Numeric evidence (NB88 metrics rows + lane-score histogram)
```
ep IoU(lane) scoreStd scoreSprd
 1   0.0718   0.0692   0.3351
 2   0.0718   0.0728   0.4630
 3   0.0718   0.0837   0.5163
[lane-score] min=0.0207 max=0.3558 ... std=0.069  (ep1)
[lane-score] min=0.0127 max=0.5290 ... std=0.084  (ep3)
```

### Diagnosis (REFUTES the cls-collapse theory)
The cls IS discriminating: score_std rising 0.069 -> 0.084, spread
0.335 -> 0.516, distribution shifting. Yet decoded IoU is byte-frozen at
0.0718. So the frozen IoU is NOT cls collapse - the GEOMETRY (curve
shape from reg_layers) is not learning while the CLS is. The rasterized
mask is constant because predicted curves stay at anchor positions
regardless of which priors the (changing) scores select.

Why the monitor "didn't work": the dead-lane monitor watches
score_std < 1e-3. score_std is 0.069 (healthy) -> correctly silent. It
was built to catch cls collapse, not frozen-IoU-with-healthy-cls. The
secondary IoU-frozen check had patience=5 (needs 6 epochs) and only ~3
epochs were visible, so it hadn't tripped.

Warnings defined:
- val.py:369 / nll_loss2d / grid_sampler: nondeterministic-algorithm
  UserWarnings from Ultralytics deterministic=True. Harmless, noisy.
- lane_mask_from_target.py:77 RuntimeWarning "invalid value in cast":
  the valid-mask `> -1e4` filtered -1e5 sentinels but let +Inf/NaN
  through into astype(int32).

### Fix
1. lane_mask_from_target.py: add np.isfinite()+range guard before int cast.
2. both train scripts: deterministic=False -> silences the 3 nondeterministic warnings.
3. epoch_callbacks.py: install_full_log_tee() (new) tees ALL stdout/stderr
   to <run>/full_train.log; drive_sync mirrors it to Drive every epoch.
   Both train scripts call it right after out_root is made.
4. epoch_callbacks.py: frozen-IoU monitor promoted to PRIMARY, patience
   5->3, loud message that says "cls learning but geometry is not - check
   reg_layers / xytl+iou weights / diff-clamp, NOT the matching scheme".

### Next diagnosis enabled
With full_train.log on Drive we can finally confirm (a) whether
'[MTDETRDLoss] lane matching scheme = hungarian' is active, and (b) add a
geometry-spread diagnostic to prove reg_layers is/ isn't moving. The
frozen geometry - not matching - is now the prime suspect for IoU.

### Verdict
Pending re-run. Note: several of these (and Phase-1 Hungarian) only take
effect once the user re-syncs the updated tree to Drive - the absence of
'[MTDETRDLoss] lane matching scheme' in the output suggests the Drive
copy was stale (same sync gap as the missing prep script).

---

## Iteration 6 - NB94: Hungarian active, IoU STILL frozen -> geometry/metric

### Symptom
NB94 (polyline + Hungarian, confirmed active) IoU frozen at 0.0710 for
all epochs while cls scoreStd rose 0.007 -> 0.072. frozen-iou-monitor
fired correctly.

### Diagnosis (cross-repo audit of CLRKDNet + RMT-PPAD)
Matching was NEVER the cause. Two compounding faults:
- Fault A: geometry (reg_layers) not learning. cls has a dense signal
  (focal over all 192 priors); reg has a sparse (Hungarian ~4/img) +
  weak (5x weight cut) + clamped (xytl diff.clamp +-100 zeros grad >100px
  off) signal. So eval preds ~= prior anchors -> rasterized mask = fixed
  anchor fan = constant -> frozen IoU. The 0.0710/0.0718 value is per-val-
  set constant = static-mask signature.
- Fault B: our metric is a hybrid neither reference uses. CLRKDNet decodes
  (conf-filter + NMS + spline) and computes curve-F1; RMT-PPAD uses a
  dense per-pixel seg head. We rasterize top-N RAW anchor preds (no NMS/
  spline) and compute pixel IoU - which can stay near-constant even if
  the model improves.

### Fix (this iteration: instrumentation + plan, NOT a blind fix)
- val.py: added [lane-geom] per-epoch logging (start_x/theta/length
  mean+std) + 4 metrics/lane_* keys. Directly tests Fault A: if the geom
  means are constant across epochs, geometry is frozen.
- PLAN_FROZEN_IOU_DIAGNOSIS.md: NB95 (is geometry frozen?), NB96 (which
  lever unfreezes reg: clamp/weights/lane-only/matching), NB97 (curve-F1
  vs pixel-IoU - is the metric lying?), NB98 (overfit 16 images). All
  short, numbered before the next full-training NB99+.

### Verdict
Root cause re-scoped from "matching" to "geometry not learning + metric
proxy". Diagnostics NB95-98 will pin which fault dominates before any
full run. Per user: diagnostic notebook numbers stay below full-training
numbers.

---

## Iteration 7 - ROOT CAUSE FOUND: rasterizer length-scaling bug

### Symptom
NB95/NB96: model trains (det R 0.08->0.42; [lane-geom] length mean
0.034->0.165 monotonic) but pixel-IoU(lane) byte-frozen (0.0500 NB96,
0.0781 NB95). Geometry MOVES, IoU does NOT -> rasterizer ignores it.

### Diagnosis (the long-hunted root cause)
`P7_validator/tools/lane_rasterize.py` lanes_to_mask:
  start  = int(round(row[2] * n_strips))   # scaled  ✓
  length = int(round(row[5]))              # NOT scaled  ✗
The PREDICTION's length row[5] is normalized [0,1] (the loss trains it
via `reg_yxtl[:,3] = pred[5] * n_strips` against the strip-count target,
so pred[5] -> ~len/71 ≈ 0.7). round(0.03..0.7) -> 0 or 1 -> `length < 2`
-> EVERY top-N prior skipped -> near-empty mask -> IoU frozen at a per-
val-set floor regardless of any training. This was frozen across NB88-96
the entire time, independent of matching/weights/clamp. The GT rasterizer
(lane_mask_from_target.py) is correct because GT row[5] is already strip
count.

### Fix
1. lane_rasterize.py: `length = int(round(row[5] * n_strips))` (match the
   start scaling + loss convention).
2. val.py: added [lane-mask] predicted lane-pixel-fraction diagnostic
   (decisive confirmation the mask is now non-empty + changing).
3. epoch_callbacks.py: reprint metrics-table HEADER every epoch (was once
   -> epoch 2+ showed bare unlabeled rows). Throttle dead-lane/frozen-iou
   monitors to fire ONCE (were 100+/run).

### Expected after fix
IoU(lane) should finally MOVE (rise from the floor as length/x-offsets
learn). [lane-mask] frac should be >0 and grow. If IoU now tracks
training, the long frozen-IoU saga is resolved and we can re-run the
NB96 ablation meaningfully + proceed to full training.

### Verdict
Pending re-run. This is the highest-confidence root cause yet: a unit
mismatch in the eval rasterizer, not matching/weights/architecture.

---

## Iteration 8 - length fix CONFIRMED + epoch-20-decline solved

### Verification (NB96, 2k img, 3 ep/row, after length*n_strips fix)
row A cut5x/clamp:   IoU 0.0034 -> 0.0140, mask-frac 0.0031 -> 0.0328
row B clrkd/noclamp: IoU 0.0049 -> 0.0203, mask-frac 0.0037 -> 0.0336
row C clrkd/dynk:    IoU 0.0044 -> 0.0193, mask-frac 0.0039 -> 0.0301
IoU now RISES every epoch (was byte-frozen). subAcc moved off 0.5000.
The frozen-IoU saga (NB88-95) is RESOLVED by the one-line length fix.

### "best at epoch 20" fix
Cause: DetMetrics fitness is detection-only -> best.pt/EarlyStopping lock
on the det peak (~ep20), ignoring the slower lane branch; then 200+
under-regularized epochs decline.
Fix: (1) val.py combined fitness = det_fit + 0.5*IoU(lane) (env
LANE_FITNESS_WEIGHT). (2) train_lane_only.py --patience=30 early stop;
full epochs 250 -> 120. best.pt preserved.

### Bezier/LCM
Appendices re-read, still viable. NB93 never run (stub). NB92 bezier rows
near-zero IoU via the separate _bezier_lanes_to_mask; needs own check
before re-run. Polyline is the validated baseline.

### Verdict
PASS: lane IoU learns. Ready for a real full-training run.

---

## Iteration 9 - overfitting analysis + augmentation fix (honest correction)

### Correction
Earlier I claimed combined-fitness + early-stopping "solved" the epoch-20
decline. They do NOT fix overfitting - they only MITIGATE the symptoms
(keep best.pt, stop wasting epochs). Owning that.

### Root-cause analysis
- Detection head = RMT-PPAD MTDETRDecoder, trained FROM SCRATCH (random
  init from YAML, no pretrained). This is IDENTICAL to the original
  RMT-PPAD train.py (it also does MTDETR(yaml) from scratch). No
  pretrained weights exist in the project.
- The real difference: original RMT-PPAD trains with DEFAULT spatial
  augmentation ON (mosaic/fliplr/scale). We set ALL spatial aug = 0
  (because lane_targets is a separate tensor that wasn't transformed with
  the image). No augmentation -> the strongest regularizer is gone ->
  detection overfits ~ep20 and val scores sit ~0.02-0.03 below the
  augmented baseline. THAT is why scores are lower AND why it overfits.

### Fix
1. augment.py RandomFlip: flip the CLR lane_targets in sync with the
   horizontal image flip (x->W-1-x, theta->1-theta; start_y/length/slots
   unchanged) using the verified P1 pixel encoding. fliplr is now
   mask-SAFE. Math hand-verified (start_x 100->539, theta 0.3->0.7,
   sentinels preserved).
2. train_lane_only.py: --fliplr (default 0.5, the strong regularizer) +
   --weight-decay (default 0.05, orthogonal). Wired into model.train.
3. NB97 config block surfaces FLIPLR/WEIGHT_DECAY.
Remaining spatial transforms (scale/translate/mosaic/perspective) stay
OFF until their target transforms are implemented.

### Verdict
Pending re-run. fliplr=0.5 should flatten the post-ep20 decline and lift
val scores. If lane IoU CRASHES with flip on, the flip encoding is wrong
-> set --fliplr 0.0 and report.

---

## Iteration 10 - NB97 success + spike diagnosis + log-truncation fix

### Findings (4 runs inspected)
- NB97 clr_lane_polyline_full_v2 (70k, clrkd+noclamp+hungarian, length fix):
  IoU(lane) 0.024 -> 0.089 (WAS frozen at 0.07). subAcc 0.54->0.68.
  mAP50 0.18 -> 0.82 still rising at ep25. **The length fix WORKS at full
  scale; lanes finally learn.** Run cut off at 25/250 (Colab timeout;
  epochs were 250 from a STALE Drive copy - my builder sets 120).
- NB92 polyline_baseline_small (10k): IoU 0.072, mAP50 0.67 (post-length-fix).
- NB92 bezier rows: IoU 0.012-0.021 (the bezier _bezier_lanes_to_mask path
  still under-draws; separate issue, deferred).

### Spikes (user's main question)
353 ll_seg spikes, up to current=27,700 vs EMA ~10. Config = clrkd
weights + diff_clamp=None.
- WHY: removing the clamp (done for NB97 after NB96's 3-ep probe) let
  reg_layers occasionally output start_x ~80 (normalized) -> x639 px ->
  smooth_l1 diff ~50800 x clrkd xytl weight 0.5 = ~25000 single-batch loss.
  The clamp (added in Iteration 1) existed exactly to bound this; clrkd's
  5x weight amplified it.
- INFLUENCE: each spike injects a huge gradient -> destabilizes the shared
  backbone -> lane geometry knocked off course -> IoU plateaued at ~0.085
  by ep10 instead of climbing. Wastes compute; risks divergence (AMP-off
  saved the 27,700 batch).
- FIX: NB97 DIFF_CLAMP 'none' -> '100' (keep clrkd). Bounds each prior's
  xytl contribution to ~50 while keeping the strong gradient.

### Log-truncation "bug" (user)
install_full_log_tee opened with 'w' -> every resume wiped full_train.log,
so a multi-session run (250 ep can't finish one Colab session) showed only
the last partial session. FIX: 'w' -> 'a' (append) + a session divider, so
resumes (FRESH=False) extend the same log. Per-epoch log_epoch_NNN.json
remain the durable metric record.

### Stale Drive
fliplr appears 0x in the NB97 log -> the run did NOT have the
augmentation/weight-decay overfitting fixes. Re-sync needed (loss.py,
val.py, augment.py, train_lane_only.py, epoch_callbacks.py, NB97).

### Verdict
PASS on the headline (lanes learn at scale, big win). Re-run NB97 with the
clamp fix (fewer spikes -> IoU should climb past 0.085) + fliplr/wd
(overfitting) once Drive is synced. Bezier path under-draw is the next
diagnostic.

---

## Iteration 11 - lane peak-then-decline = negative transfer (gate-proven)

### Symptom
User stopped NB97: IoU(lane) peaks ep11 (0.0915) then declines to 0.085;
subAcc same. Asked if it's the spikes.

### Diagnosis (gate trajectory is the smoking gun)
NB97 (stale config: clamp=None, no fliplr). Lane IoU+subAcc peak ep11,
decline; mAP50 KEEPS rising 0.18->0.82. GCA gates: Det_gate 0.51->0.65
(detection decouples), Seg_gate 0.51->0.54 (lane STAYS coupled to shared
trunk). => multi-task NEGATIVE TRANSFER: the shared backbone keeps being
optimized for detection and overwrites lane features after the lane peak.
The spikes add noise but are NOT the primary cause. pixel-IoU is also a
weak proxy (saturates ~0.09) - true quality needs curve-F1.

### Fix (targeted)
train_lane_only.py: new --freeze-trunk-after N. At epoch N, a
on_train_epoch_start callback freezes the shared backbone+neck
(model[0:28], also .eval() for BN) so only the task decoders + GCA + lane
head train on a locked trunk. Stops the trunk drifting to detection ->
lanes can keep climbing. Default -1 (off). Plus PLAN_LANE_PEAK_DECLINE.md.

### Honest status
The earlier clamp/fliplr/early-stop fixes were NOT in the declining run
(stale Drive) and are mitigations, not the cure. The freeze knob is the
first targeted fix. Plan: Exp1 re-run synced (clamp+fliplr), Exp2
--freeze-trunk-after 12, Exp3 curve-F1 (know the truth), Exp4 GCA.

### Verdict
Diagnosis solid (gate-proven). Fix implemented but UNTESTED. best.pt
already preserves the ep11 peak regardless.
