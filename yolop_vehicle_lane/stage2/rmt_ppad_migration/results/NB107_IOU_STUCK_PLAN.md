# NB107 — "F1 rising, lane-IoU stuck": diagnosis + resume plan

Date 2026-06-02. For execution in the NEXT conversation. The run is
`laneiou_both_full` (Sprint-2 winner: LaneIoU loss+matcher + aux + gca + tau0.4,
fitness=curvef1), 70k/120ep, stopped at **epoch 18**.

## 1. The data (epochs 1 → 18)

| metric | ep1 | ep5 | ep10 | ep18 | shape |
|---|---|---|---|---|---|
| **lane_f1@0.5** | 0.001 | 0.427 | 0.562 | **0.608** | ✅ rising, not plateaued |
| **curveIoU** | 0.407 | 0.641 | 0.706 | **0.735** | ✅ rising |
| **pixel-IoU(lane)** | 0.048 | 0.066 | 0.056 | **0.054** | ⚠️ FLAT ~0.05 (the "stuck") |
| mask_frac | 0.029 | 0.030 | 0.028 | 0.028 | flat ~0.028 |
| length_mean | 0.155 | 0.102 | 0.098 | **0.094** | ⚠️ shrinking |
| precision / recall | .03/.00 | .54/.35 | .60/.53 | **.625/.591** | ✅ both rising, balanced |
| mAP50 (guardrail) | 0.14 | 0.75 | 0.81 | **0.823** | ✅ healthy |

## 2. Diagnosis — the "stuck IoU" is NOT a failure; it is the WRONG METER

**The headline metric (curve-F1) is the BEST in the project: 0.608 and still
climbing** — it already clears the Sprint-2 Gate-G2 target of 0.45, on full data.
curveIoU 0.735 is the highest ever. precision (0.625) and recall (0.591) are
balanced and rising — the over-prediction / "always 8 lanes" problem is GONE.
mAP50 holds 0.82. **LaneIoU + the curvef1 fitness worked.**

What is "stuck" is **pixel-IoU(lane) ~0.05 and mask_frac ~0.028**. Both are the
SAME saturating thin-line metric we have flagged all project: a merged binary
mask of ~3px-wide predicted lines vs ~3px GT lines saturates near ~0.05-0.09
regardless of how good the curves are. It is a DEAD proxy here — note F1 went
0.001 → 0.608 while pixel-IoU sat at ~0.05. They are measuring different things;
F1/curveIoU are the faithful ones (per the NB104 saturation analysis), which is
exactly why we switched the fitness off pixel-IoU.

**BUT there is one real, fixable signal hiding inside the "stuck": `length_mean`
is SHRINKING (0.155 → 0.094) and mask_frac with it.** The model is learning to
predict SHORTER lanes — it covers the easy mid-section and gives up on the
extremities (especially near-field). Shorter predicted lines => less drawn mask
=> pixel-IoU and mask_frac stay low EVEN as the drawn part gets more accurate
(F1 up). This is the documented "length shrink" failure (NEXT_STEPS S1.4), and
it is the ONE thing actually worth fixing in the IoU signal.

Root cause of the length-shrink: the LaneIoU loss (correctly) rewards overlap on
the rows the lane covers, and the cls/threshold rewards precision — but NOTHING
strongly penalizes a prediction for being too SHORT. The xytl smooth-L1 on the
`length` field (reg_yxtl[:,3]) is one term among four and is clamp-limited; the
model finds it cheaper to shorten than to nail the hard near-field geometry.

## 3. Plan — resume from ep18, add length supervision (do NOT restart)

The user wants to RESUME, not restart. Two changes, both apply on resume:

### Step A — `--resume` flag — DONE THIS CONVERSATION (wired + unit-tested)
`train_lane_only.py` now has a `--resume` arg (default `''` = fresh run). Pass a
checkpoint path, or `'auto'` for `{project}/{name}/weights/last.pt`. The
`model.train(...)` call now forwards `resume=_resume_arg`. Verified against the
vendored `trainer.check_resume()`/`resume_training()`:
- It restores optimizer + epoch + best_fitness + EMA from the `.pt` and runs
  `start_epoch(19) -> epochs(120)`. cos-LR + EarlyStopping(patience=30) state
  ride along. (assert fires only if the ckpt run already *finished* — ours is
  ep18/120, safe.)
- `check_resume` does `self.args = get_cfg(ckpt_args)` — it REPLACES all args
  with the checkpoint's saved hyperparameters. Only `imgsz/batch/device/epochs`
  from the new CLI still override. So lr0/patience/weight_decay/fliplr/etc.
  revert to the original NB107 recipe — which is exactly what we want for a
  faithful continuation.
- The lane levers are env vars read at `MTDETRDLoss.__init__` (loss-construct
  time), NOT ultralytics args — so **Step B's `--lane-y-reweight near` DOES take
  effect on resume** even though the args.yaml is reloaded. (Confirmed the code
  path; this is the whole reason the lever can be toggled mid-run.)
- Resolution logic unit-tested (4 branches: off/auto/explicit-existing/
  explicit-missing-raises). py_compile clean.

**CRITICAL Colab logistics (the #1 resume risk):** `/content/` is WIPED between
sessions. The original run's `last.pt` lives only on Drive at
`training_runs/checkpoints/laneiou_both_full/last.pt` (drive_sync writes it each
epoch). In a fresh session `{project}/{name}/weights/last.pt` does NOT exist, so
`--resume auto` will (correctly) raise FileNotFoundError. Two safe options:
  1. Pass the Drive path directly: `--resume /content/drive/MyDrive/<...>/`
     `laneiou_both_full/last.pt`. Simplest; resume loads straight from Drive.
  2. Restore the whole run dir Drive→`/content/runs/laneiou_full/`
     `laneiou_both_full/` first (so results.csv continuity + ultralytics
     bookkeeping are intact), then `--resume auto`.
Prefer (2) for clean continuous results.csv/monitoring; (1) is the minimal path.
Note ultralytics reloads `save_dir` from the ckpt args, so resumed epochs write
to the ORIGINAL save_dir (recreated if absent) and drive_sync re-mirrors — no
config drift.

### Step B — turn ON the length-anti-shrink lever (S1.4 y-reweight 'near')
We already BUILT and unit-tested `LANE_Y_REWEIGHT=near` (the near-field
up-weight) for exactly this symptom; it was OFF in NB107. On resume, set
`--lane-y-reweight near`. It up-weights the IoU loss on the near (bottom) rows
where the lane is being abandoned, so shortening becomes costly there. Expected:
length_mean stops shrinking / recovers, mask_frac + pixel-IoU tick up, and
near-field geometry (the NB103 sawtooth zone) improves — WITHOUT hurting F1
(it is the same LaneIoU loss, just row-weighted).

### Step B2 — DIRECT length hinge — IMPLEMENTED (promoted from fallback)
UPDATE after reading the code: Step B (y-reweight) is INDIRECT — it reweights
the IoU loss on x-offsets (cols 6+), which is masked by GT-TARGET validity
(lane_losses.py:50,61), so it sharpens near-field x-accuracy but NEVER
supervises the `length` field (col 5) that is actually shrinking. The length
field is supervised only by the xytl smooth-L1, which is SYMMETRIC and is 1/4 of
a mixed-scale mean (start_x ≤639px, theta ≤180°, length ≤71) — so it neither
biases against shortening nor weighs length much. THAT is the shrink mechanism,
and y-reweight does not touch it.

So the length hinge is not a fallback; it is the primary DIRECT fix, now built:
- `loss.py`: `LANE_LEN_HINGE_W` lever → `xytl_loss_sum += w * ((gt_len -
  pred_len).clamp(min=0)/n_strips).mean()` on the MATCHED priors only (asymmetric
  — penalizes too-short, ignores too-long; the symmetric smooth-L1 still caps
  over-long). Off by default (w=0).
- `train_lane_only.py`: `--lane-len-hinge` flag → sets the env var.
- Unit-tested (test_sprint1_levers.py test 5: short→+, exact→0, long→0, scales
  with w). Regression suite 6/7 (unchanged baseline). No vendor-shim change
  needed (the hinge is inline in MTDETRDLoss, uses existing tensors).
- NB108 runs `--lane-len-hinge 2.0` alongside `--lane-y-reweight near`. 2.0 is a
  starting estimate; tune up (3-4) if length doesn't recover, down if F1 dips.

METRIC CAVEAT: `length_mean` (val.py:509) averages ALL 192 priors, not just
matched lanes, so it's a noisy proxy (background priors → ~0 length pull it
down). The hinge targets MATCHED lanes correctly regardless. Watch
**pixel-IoU(lane) + mask_frac rising** (cleanest "more lane drawn" signal) and
curve-F1/curveIoU holding, over length_mean.

### Step C — re-judge on the RIGHT metric
Stop reading pixel-IoU as the lane-quality meter. Track **curve-F1 + curveIoU +
length_mean + mAP50**. Promotion/G2 already PASSED on F1 (0.608 ≥ 0.45). The
resume is to (a) finish the schedule to ~ep40-60 and see where F1/curveIoU peak,
and (b) confirm Step B recovers length_mean (the only real IoU-signal defect).

## 4. What NOT to do
- Do NOT restart from scratch (user explicit; and the run is the project best).
- Do NOT chase pixel-IoU ~0.05 as a target — it is saturated/dead here.
- Do NOT re-enable S1.3 iou_cls (NB105: collapses score_std → F1=0).
- Do NOT touch the LaneIoU/matcher config — it is working (F1 0.608).

## 5. Next-conversation execution order
1. ~~Add `--resume` flag~~ DONE this conversation (wired + unit-tested; see §3A).
2. Build NB108 (resume notebook). Reuse the NB107 builder/recipe verbatim — same
   model YAML, 70k prep, aux+gca, `--lane-iou-type laneiou --lane-iou-match
   laneiou --lane-eval-tau 0.4`, fitness=curvef1 — and add exactly two deltas:
   - restore `last.pt` from Drive (option 2 in §3A: copy the run dir
     Drive→/content first), then
   - the new flags: `--resume auto --lane-y-reweight near`
   (or `--resume <drive-last.pt-path> --lane-y-reweight near` for the minimal
   path). Everything else identical so the only changes vs NB107 are
   resume + the near-field reweight.
3. Sanity-check the launch log shows: `RESUME from <...>/last.pt` and
   ultralytics `Resuming training ... from epoch 19 to 120 total epochs`. If it
   prints epoch 1, the resume did NOT take — stop and fix before burning GPU.
4. Run ~20-30 more epochs. Watch **length_mean recover** (target: stop the
   slide, ideally climb back toward ~0.12+) and **F1/curveIoU keep rising or
   hold**. mAP50 must stay ≥0.48 (guardrail). If length_mean is still <~0.12 by
   ~ep28, add Step B2 (length hinge / raise xytl weight).
5. Make the final ship / Sprint-3 (curve-query DETR) call on **curve-F1**, not
   pixel-IoU. F1 already cleared G2 (0.608≥0.45); the resume is to find the peak
   and confirm the length fix.

## 6. Status of deliverables
- [x] Inspected NB107 results.csv (18 epochs) — full trajectory in §1.
- [x] Diagnosis written (§2): "stuck IoU" = saturated thin-line metric; the real
      defect is length-shrink (length_mean 0.155→0.094).
- [x] Plan written (§3): resume + turn on the existing `near` y-reweight lever.
- [x] `--resume` mechanism wired into train_lane_only.py + unit-tested (§3A).
- [x] Verified the `near` reweight is REAL (loss.py `_y_reweighted_liou`, 1.85×
      near rows, replaces not doubles the base IoU term, composes with the
      angle-aware LaneIoU `lane_iou_loss_fn`). Backed by passing
      test_sprint1_levers.py + test_lane_iou.py (regression suite 6/7, the 1
      fail is the pre-existing lane_rasterize junk-fixture).
- [x] **DIRECT length hinge implemented** (§3 Step B2): `LANE_LEN_HINGE_W` lever
      in loss.py (asymmetric too-short penalty on matched lanes) + `--lane-len-
      hinge` flag + unit test (test_sprint1_levers test 5). This was promoted
      from "fallback" after the code showed y-reweight is indirect.
- [x] **NB108 built**: `notebooks/nb108_build_helper.py` →
      `stage2_notebook_108_laneiou_resume_yreweight.ipynb`. Reuses NB107's
      dataset-prep cells verbatim (imported, no drift), adds Cell 3c (restore
      last.pt from Drive) and swaps Cell 4 to `--resume auto --lane-y-reweight
      near --lane-len-hinge 2.0`. JSON + all 7 code cells validated.
- [ ] RUN on Colab (user step): execute NB108 cells 1→5. Confirm Cell 4 log
      prints "Resuming training ... from epoch ~19 to 120" (NOT epoch 1), then
      watch length_mean recover while curve-F1/curveIoU hold/rise and mAP50≥0.48.
