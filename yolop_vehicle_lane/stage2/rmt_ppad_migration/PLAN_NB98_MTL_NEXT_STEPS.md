# NB98 MTL ablation — results, findings, and next-steps plan

Date 2026-05-31. All 7 rows trained to completion (20 ep). For execution
in the NEXT conversation.

## The numbers (read from each `results.csv`)

| Row | IoU(lane) ep1→ep10→ep20 | curveIoU (all ep) | lane_f1 | mAP50 ep1→ep20 |
|---|---|---|---|---|
| `abl_baseline`  | 0.0165 → 0.0350 → 0.0471 | **0.0000** | 0 | 0.036 → 0.697 |
| `abl_asym_lr`   | 0.0133 → 0.0319 → **0.0346** | 0.0000 | 0 | 0.008 → **0.424** |
| `abl_gca_floor` | 0.0169 → 0.0377 → **0.0583** | 0.0000 | 0 | 0.032 → 0.702 |
| `abl_det_decay` | 0.0145 → 0.0406 → 0.0542 | 0.0000 | 0 | 0.029 → 0.686 |
| `abl_freeze`    | 0.0163 → 0.0382 → 0.0417 | 0.0000 | 0 | 0.029 → 0.643 |
| `abl_pcgrad`    | 0.0152 → 0.0396 → 0.0567 | 0.0000 | 0 | 0.030 → **0.709** |
| `abl_aux_seg`   | 0.0141 → 0.0444 → **0.0642** | 0.0000 | 0 | 0.028 → 0.701 |

## Finding 1 (BLOCKER) — `curveIoU` reads exactly 0.0000 everywhere → metric wiring bug

`metrics/lane_curveIoU(lane)` is **0.0000 for all 7 rows × all 20 epochs**,
yet `IoU(lane)` (merged pixel-IoU) is nonzero and rising and `lane_mask_frac`
~0.03–0.06 (lanes ARE drawn). NB99 already PROVED the curve-F1 function is
correct in isolation (perfect match → curveIoU≈0.96; 192-prior → 0.76). So
exactly-zero with zero variance is **not** a weak model — it is a wiring bug
in the val loop, almost certainly **GT decodes to empty (n_gt=0)**: then
`lane_curve_tp_fp_fn` returns `iou_sum=0, n_gt=0` and get_stats emits
`curveIoU = 0`. (If preds also failed we'd get an empty column, not 0.0; we
get 0.0, so preds decode—fp>0—but GT does not.)

**=> The headline metric for this whole ablation is currently unusable. Fix
it before judging techniques on the un-saturated metric. `lane_f1=0` is also
uninformative (same root or genuinely sub-threshold).**

## Finding 2 — on the pixel-IoU FALLBACK, there is NO post-ep10 decline

Every row's `IoU(lane)` RISES monotonically ep1→ep20 (baseline included:
0.0165→0.0350→0.0471). **The negative-transfer "peak-then-decline after
ep5" that motivated this whole suite did NOT reproduce on the 20-ep/10k
regularized recipe** — consistent with the earlier NB92/NB97 finding that
clamp+fliplr+wd already turned the decline into a plateau/rise. So the suite
is really measuring "which technique lifts lane IoU," not "which stops a
decline" (there's no decline to stop here).

## Finding 3 — provisional ranking (pixel-IoU, since curveIoU is broken)

- **Best lane IoU: `abl_aux_seg` (0.0642) > `abl_gca_floor` (0.0583) >
  `abl_pcgrad` (0.0567) > `abl_det_decay` (0.0542)** — all keep mAP50 ~0.70.
  Dense aux supervision and feature-decoupling (gca floor) help most.
- **`abl_asym_lr` HURT badly:** lowest lane IoU (0.0346) AND mAP50 collapsed
  to 0.42 — backbone LR ×0.1 starved BOTH tasks. Drop it (or retry ×0.3).
- **`abl_freeze` underperformed:** capped lane IoU (0.0417) and mAP50 (0.64)
  — freezing at ep10 just stopped both from improving (there was no decline
  to prevent, so it's net-negative on this budget).
- **`abl_pcgrad` ran cleanly** (no NaN, mAP50 0.71) — the risky one worked,
  but its lane gain over baseline is modest.

Caveat: pixel-IoU saturates ~0.09, so these gaps (0.047 vs 0.064) are near
the noise floor of a weak metric. **Confirm on the fixed curveIoU.**

---

## PLAN (next conversation, step by step)

### Step 1 — FIX the curveIoU val-loop wiring (priority 1, do first)
1. Instrument the curve-F1 block in `vendor/.../mtdetr/val.py` update_metrics:
   for the first val image of the first batch, log `n_gt, n_pred` and, if
   `n_gt==0`, dump one `batch['lane_targets'][si]` row (slots 1,2,5 + a few
   x's) so we see why `_decode_lane_polyline(is_pred=False)` rejects it.
2. Cross-check against the PROVEN GT decoder
   `P6_dataset/tools/lane_mask_from_target.py` (it decodes the same tensor
   into the GT mask pixel-IoU uses, so it is correct). DIFF the two on one
   real row — the curve-F1 GT decode must match it (validity slot, length as
   strip-count, x as 640-px → scaled by R/640).
3. Verify: load any row's `best.pt` (on Drive) and run a 1-batch val; assert
   `curveIoU > 0`. Then re-confirm NB99 still passes.

### Step 2 — re-judge the ablation on the FIXED curveIoU
Re-run val (or re-read results if curveIoU now populates on a re-run) and
rank by curveIoU. Pixel-IoU says aux_seg/gca_floor lead — confirm or correct
on the honest metric.

### Step 3 — promote the real winner(s)
- Combine the top 2 non-conflicting techniques (likely
  `--use-aux-seg` + `--seg-gate-floor 0.3`) on the same 10k/20ep budget.
- If confirmed, launch a full-data NB97-style run (120 ep + early stop) with
  the winner folded into the polyline recipe. Watch curveIoU as headline.

### Step 4 — the precision wall (likely the true bottleneck)
`lane_f1=0` everywhere and IoU plateaus ~0.06. Even once curveIoU works it
will likely be modest. Cheap→targeted: (a) higher-res lane rasterization in
loss/eval (R 160→320); (b) raise xytl/LineIoU weight vs cls; (c) track
curve-F1 at IoU 0.3 too; (d) train the winner longer (lanes still rising at
ep20).

### Step 5 — prune the losers
Drop `asym_lr ×0.1` (starves both tasks) — optionally retry ×0.3. Deprioritize
`freeze` (no decline to prevent on this budget). Keep `pcgrad` only if the
fixed curveIoU shows a real edge (it's 2× backward cost).

### Step 6 — bezier track (parallel, see BEZIER_LCM_POSTMORTEM_AND_PLAN.md)
t-range fix verified; next is I1 (curve-sampling loss) + I2 (flip aug), then
re-test LCM. Fold the winning MTL technique into the bezier trainer.

### Cleanup
Remove `training_runs/checkpoints/_parse_abl.py` + `_abl_report.txt` once the
ranking is in `debug_record.md`.

## What NOT to do
- Don't rank techniques on curveIoU until Step 1 fixes it (it's 0 for all).
- Don't read the absence of decline as "techniques failed" — there was no
  decline to fix; judge them on lane-IoU/curveIoU lift instead.
- Don't promote to a full-data run before Step 2 confirms a winner on the
  fixed metric (avoid a ~7 GPU-h run on a pixel-IoU-noise-level difference).
