

---

## 2026-05-31  Local regression suite + autonomous verification (no GPU)

**Context:** built NB101 (combined-winner full training: aux_seg+gca_floor) with
a pre-train DA+polyline+bbox inspection cell, NB102 (build complete 3-modality
dataset -> Drive using the user-provided `bdd100k_drivable_maps.zip`), and
hardened `extract_drivable_subset.py` (prefer id-mask over RGB colormap).

**Verification (executed locally with numpy/cv2/torch-cpu, results in
`tests/RESULTS.txt`):** added a permanent CPU-only regression suite under
`tests/` and ran it green **5/5**:
- `test_lane_rasterize` - PRED `length*n_strips` fix + PRED/GT rasterizers agree
  (gt_px == pred_px == 5174 for the same line).
- `test_curve_f1` - broad-pool curveIoU (top-64) + preds not gated by row[1]<0.5;
  pool=8 curveIoU~0 vs pool=64 >0.5 on the low-logit-match case; bezier path OK.
- `test_drivable_extract` - synthetic BDD "Drivable Maps" zip end-to-end: id-mask
  preferred over colormap, 0/1/2->binary, coverage 4/4+3/3, no-label stem
  correctly reported missing (NOT fabricated). This is NB102's critical path.
- `test_notebook_decode` - NB101 & NB102 inline `decode_lanes`/`read_boxes` match
  the canonical rasterizer (inspection overlays are correct).
- `test_bezier_trange` - `render_with_validity` honors the range (full 0.02-0.98
  span=0.800 vs 0.50-0.55 stub), validating the bezier t-range init fix.

**Env note:** the sandbox tool-display layer garbled multiline stdout AND file
reads this session (phantom "...", "[truncated]", duplicated JSON keys). Ground
truth was obtained via exit codes + python-written result files + Read of those
files. python's own file I/O was unaffected (all tests operate on real files).

**Verdict:** the production-critical data paths (polyline lane rasterize, the
curveIoU/F1 metric, drivable extraction, notebook inspection overlays) are
verified by execution, not just static analysis. Re-run anytime with
`python tests/run_all.py`.


---

## 2026-06-01  Lane-head plateau: adopt NEXT_STEPS strategy, Sprint 0 (NB104)

**Correction to prior turn:** I called combined_aux_gca_full "not stuck" by
leading with F1 0.05->0.30. The FULL csv shows F1 PEAKS 0.244@ep9 then FLAT
0.236-0.237 through ep16 - the user was right: lane head plateaued ~5 epochs.

**User NB103 observations (decisive qualitative evidence):**
- predicted lanes roughly align with GT but shapes differ a lot;
- far-field lanes smooth, NEAR-field / center lanes jagged/sawtooth, swaying L-R;
- model always emits 8 lanes even when GT has ~0.

**Diagnosis (per results/NEXT_STEPS_lane_head_strategy.md, grounded in csv+code):**
two root causes, not six. R1 = representation: 78-D per-row polyline, 72 x-offsets
regressed independently with no inter-row smoothness prior + tolerant LineIoU ->
near-field angular noise becomes large per-row x noise = sawtooth; caps curveIoU
~0.09, F1 ceiling ~0.236. R2 = set prediction: classifier never sharpens
(score_std~0.09) AND decode takes top-8 with NO score threshold -> structurally
cannot output 0 lanes. The MTL/negative-transfer axis (NB94-101) is near-exhausted
(mAP50 holds ~0.50, trunk healthy); gains are now lane-head-internal.

**Action this turn (Sprint 0, NO training):**
1. FIXED the bezier metric decoder bug (S0.1): `_decode_lane_bezier` in
   P7_validator/tools/lane_curve_f1.py gated PRED on row[1]<0.5 (a cls LOGIT,
   routinely <0.5) -> pinned bezier curve-F1 at 0 for all 30 epochs => bezier runs
   were MIS-SCORED/abandoned prematurely. Now only gates GT (mirrors polyline).
   Verified: low-score bezier pred decodes (72 pts, was None); GT invalid still
   dropped. Decoder now unified across polyline+bezier.
2. Built NB104 = Sprint 0 instrument notebook (no GPU training): S0.1 re-score
   bezier ckpts with fixed decoder; S0.2 score histogram + separability AUC; S0.3
   threshold sweep (precision/recall/F1 + pred-count-vs-GT-count corr); S0.4
   near-vs-far jaggedness (mean-sq 2nd-diff of x). Produces the 4 exit numbers
   that gate Sprint 1. Verified: 17 cells, 8 code, parse clean, single decoder.

**Next:** user runs NB104 -> 4 numbers -> then NB105 = Sprint 1 ablation grid
(threshold / smoothness reg / IoU-aware cls / y-reweight, 10k/15ep). Targets:
F1>=0.32, near-jaggedness halved, pred-count tracks GT, mAP50>=0.48 guardrail.
Plan allows swapping CLRKDNet head for CLRerNet/Bezier/GANet/curve-query DETR in
Sprint 2/3 if representation is the binding wall.


---

## 2026-06-02  Sprint-0 instrument (NB104) — the four gating numbers

Ran NB104 on combined_aux_gca_full/best.pt, 150 val images, FIXED decoder.

- S0.1 bezier curve-F1 (re-scored, fixed decoder): bezier_cubic=0.105 vs
  polyline_NB101=0.305. Bezier was NOT mis-scored — genuinely ~3x worse.
  => DO NOT resurrect bezier (kills Sprint-2 S2.B). (lcm row failed to load:
  shim name 'b5_p5_lane_losses' unregistered — cosmetic, ignored.)
- S0.2 cls separability AUC = 0.960 (matched mean 0.361 vs unmatched 0.113).
  Highly separable => a confidence THRESHOLD works; IoU-aware cls (S1.3) is
  NOT mandatory.
- S0.3 best decode-only F1 = 0.346 @ tau=0.4 (vs 0.305 at fixed 8 lanes),
  +13% FOR FREE. cnt_corr 0.00->0.64 and pred_cnt 8.0->4.7 (GT~4.9): the
  threshold FIXES the "always-8-lanes" cardinality. => S1.1 thresholded
  decode is the big cheap win, do first.
- S0.4 jaggedness NEAR=758 vs FAR=1533 px^2, ratio 0.49x. SURPRISE: near is
  SMOOTHER than far, opposite of the NB103 visual read. Likely the NB103
  "center jaggedness" was multiple OVERLAPPING false-positive lanes (cured by
  S0.3 threshold), not one jagged curve. => smoothness-reg (S1.2) is LOWER
  priority than thought; re-judge after thresholding.

DECISION: Sprint 1 leads with S1.1 (threshold, free +13% + cardinality fix),
keeps S1.3 (IoU-aware cls) as a refinement not a necessity, demotes S1.2
(smoothness) to an optional row pending a post-threshold re-look. Bezier track
closed.


---

## 2026-06-02  Sprint-1 ablation (NB105) — IMPLEMENTATION BUG found, not a finding

10k/15-ep probes, all eval @ tau=0.4. Results (final epoch):
  row            F1     score_std  mAP50
  s1_baseline    0.148  0.082      0.671   <- the bar (15ep from scratch)
  s1_iou_cls     0.000  0.005      0.683   <- BROKEN
  s1_yreweight   0.043  0.082      0.677   <- hurt (traded away far-field)
  s1_iou_yrw     0.000  0.005      0.686   <- BROKEN
  s1_all_smooth  0.000  0.005      0.692   <- BROKEN

ROOT CAUSE (mine): the S1.3 QFL soft target = raw line-IoU of the matched prior.
Early in training that IoU is ~0.05 (matches pixel-IoU), so QFL drives EVERY
positive score toward ~0.05 -> all scores collapse to near-0 -> score_std
0.082->0.005 -> the tau=0.4 threshold rejects EVERY lane -> F1=0. Verified by a
QFL-vs-target sweep: soft_target=0.05 minimizes at pos-prob 0.047.
My unit test passed only because it used target=1.0/0.5, never the realistic
~0.05 regime. Test was right on math, wrong on regime.

mAP50 held 0.67-0.69 on ALL rows (>=0.48 guardrail OK) -> no negative transfer;
this is purely a lane-cls-target scaling bug.

Secondary: s1_yreweight (cls healthy) HURT F1 0.148->0.043 -> near-weighting
traded away far-field that was scoring. y-reweight 'near' is too aggressive.

FIX OPTIONS for S1.3 (next): (a) rescale the soft target = IoU / running_max_IoU
(or a fixed /0.3) so a good-geometry prior targets ~1, not ~0.05; (b) blend:
target = max(0.5, IoU) for matched priors (keep a strong positive signal while
encoding quality); (c) QFL only on the POSITIVE-vs-its-own-quality, keep hard 1
floor. Re-test with the low-IoU regime in the unit test BEFORE another GPU run.
NB104 S0.3 thresholded F1=0.346 was on the 120ep NB101 model, not 15ep probes,
so 0.148 baseline @15ep is the correct comparison bar.


---

## 2026-06-02  Sprint-1 ablation (NB105) — G1 FAILED; escalate to Sprint 2

NB105, 10k/15ep. Baseline REUSED from round 1 (loss byte-identical with levers
off, so re-running was unnecessary).

| row | F1 | score_std | mAP50 |
|---|---|---|---|
| s1_baseline (reused) | 0.148 | 0.082 | 0.671 |
| s1b_iou_cls (QFL g=2, floor=0.5) | 0.000 | 0.022 | 0.698 |
| s1b_iou_mildyrw (+angle) | 0.000 | 0.024 | 0.694 |

- S1.3 IoU-aware cls HURTS, AGAIN. score_std COLLAPSED 0.082->0.022 (histogram a
  single spike at 0; max score 0.31). At tau=0.4 every lane is rejected -> F1=0.
- WHY my round-1 "fix" (floor=0.5,norm=0.3 soft-target map) did not work: the
  unit test validated the TARGET MAPPING in isolation, but the real failure is
  the ~40:1 negative:positive prior imbalance. QFL's |t-p|^gamma modulation does
  NOT counter that imbalance the way the original FocalLossForLane(alpha=0.25)
  did, so the classifier drives ALL priors toward 0. The test missed this
  because it never exercised the imbalance / training dynamics. Honest limit of
  helper-level testing.
- S1.4 y-reweight: un-judgeable (rode on the broken iou_cls rows). Given S0.4
  (near already smoother than far), not worth isolating.
- S1.1 threshold remains the ONLY S1 win (free, banked). Baseline F1=0.148 is the
  Sprint-1 ceiling, far below G1 target 0.32.

GATE G1: FAILED. Two independent rounds show cheap cls/decode tweaks cannot break
the plateau, and S0.2 already showed the classifier RANKS fine (AUC 0.96) -- the
problem is GEOMETRY, not calibration. Per the plan's decision tree -> the
REPRESENTATION is the binding constraint -> Sprint 2 (S2.A CLRerNet LaneIoU).

ABANDON S1.3 (revert iou_cls path / leave it env-gated OFF). Keep S1.1 threshold.

Sprint-2 readiness: external_repos/CLRerNet-main present. Core is LaneIoULoss
(libs/models/losses/iou_loss.py): angle-aware virtual lane width =
lane_width*sqrt(dx^2+dy^2)/dy -- widens the IoU band where the lane is steep
(near-field, our jaggedness zone), so it penalizes shape error proportional to
local angle. Our current line_iou uses a FIXED band (length=15) that
under-penalizes exactly there. Pure tensor ops, no mmdet at runtime -> ports
into our line_iou/liou_loss cleanly. THIS is the S2.A change.


---

## 2026-06-02  NB106 crash fix — vendor lane_losses shim missing lane_iou export

NB106 both training rows crashed at model build:
  ImportError: cannot import name 'lane_iou_loss' from
  vendor/RMT-PPAD/ultralytics/models/utils/lane_losses.py
Root cause: S2.A added lane_iou/lane_iou_loss to the P5 SOURCE
(P5_loss/tools/lane_losses.py) but the VENDOR SHIM re-exports each name
explicitly (line_iou=_p5.line_iou, ...) and its list + __all__ were not updated.
loss.py imports from the shim -> missing name -> crash before training.
FIX: added `lane_iou = _p5.lane_iou` + `lane_iou_loss = _p5.lane_iou_loss` and
both to __all__ in the vendor shim. Verified the full import chain resolves
(shim -> P5 -> both names) and the regression suite stays green (6/7; the 1 fail
is the pre-existing test_lane_rasterize junk-fixture, lane_rasterize.py untouched).
Lesson (recurring): when adding a function to a P-source, ALSO add the re-export
line to its vendor shim. This is the same class as the earlier
p5_lane_losses / b5_p5_lane_losses pickle-name misses.
NB106 is otherwise unchanged and ready to re-run after re-sync.


---

## 2026-06-02  Sprint-2 LaneIoU (NB106) — curveIoU UNFROZE; matcher is the key

NB106, 10k/15ep, full aux_seg+gca_floor+hungarian+tau0.4 recipe; only IoU type
varies. Baseline reused from NB105.

| row | F1@0.5 | curveIoU | mAP50 | scoreStd | subAcc |
|---|---|---|---|---|---|
| s2_baseline (line, reused) | 0.148 | 0.517 | 0.671 | 0.082 | -- |
| s2_laneiou_loss (loss only) | 0.049 | 0.600 | 0.680 | 0.076 | 0.589 |
| s2_laneiou_both (loss+matcher) | **0.262** | **0.610** | 0.673 | 0.073 | 0.632 |

KEY FINDINGS
- curveIoU UNFROZE: 0.517 -> 0.61 (both LaneIoU rows). The long-stuck geometry
  signal finally moved -- angle-aware width helps the geometry. pixel-IoU(lane)
  also rose 0.06 -> 0.091 (both), subAcc 0.589 -> 0.632.
- THE MATCHER IS THE LEVER, NOT THE LOSS. LaneIoU in the LOSS ONLY = F1 0.049
  (P 0.54 / R 0.026 -- precise but finds almost nothing: the LineIoU matcher
  assigns the wrong priors, so few survive the tau=0.4 top-8). LaneIoU in BOTH
  loss + matcher = F1 0.262, +77% over the 0.148 baseline (P 0.48 / R 0.18).
  => the angle-aware ASSIGNMENT (geometry-faithful matching) is what lifts F1;
  the loss term alone is counter-productive without it. This is exactly
  CLRerNet's "use LaneIoU for both" recipe, now confirmed on our model.
- mAP50 held 0.67-0.68 across all rows (guardrail OK; no negative transfer).

GATE G2: PARTIAL. curveIoU>=0.25 PASS (0.610), mAP50>=0.48 PASS, but F1>=0.45
NOT met (0.262). Per the plan: curveIoU clearly off the 0.09 floor => LaneIoU
is working and the representation is NOT fully exhausted -- promote s2_laneiou_both
to a LONGER/full run before judging Sprint 3. The 15-ep probe likely undershoots
F1 (baseline polyline needed ~ep9 to peak; LaneIoU may need more).

DECISION: (1) promote s2_laneiou_both to 70k full-data (or >=40ep 10k) to see if
F1 clears 0.45 with more training; (2) hold Sprint 3 (curve-query DETR) until
that longer run is judged -- G2 is partial, not failed.

================================================================================
NB107  laneiou_both_full  70k/120ep  (the promoted full run)  2026-06-02
================================================================================
Stopped at ep18. results.csv trajectory (ep1 -> ep18):
  lane_f1@0.5   0.001 -> 0.608   (rising, NOT plateaued -- project best)
  curveIoU      0.407 -> 0.735   (rising -- highest ever)
  mAP50         0.139 -> 0.823   (healthy; guardrail OK)
  precision     0.033 -> 0.625 | recall 0.001 -> 0.591  (balanced, rising;
                the "always 8 lanes" over-prediction is GONE)
  pixel-IoU(lane) 0.048 -> 0.054 | mask_frac 0.029 -> 0.028  (FLAT -- "stuck")
  length_mean   0.155 -> 0.094   (SHRINKING)

USER REPORT: "F1 increasing but lane IoU stuck." DIAGNOSIS: the stuck pixel-IoU
is the WRONG METER -- it's the same saturating thin-line mask metric (~0.05-0.09
ceiling for 3px lines) we switched the fitness OFF of. The faithful metrics
(curve-F1 0.608, curveIoU 0.735) are the best in the project and still climbing;
G2's F1>=0.45 is now CLEARED on full data. The one REAL signal inside the
"stuck" is length_mean shrinking 0.155->0.094: the model draws shorter lanes
(covers the easy mid-section, abandons the near/far extremities), so the drawn
mask stays tiny even as the drawn part gets more accurate. = the documented
S1.4 length-shrink failure; nothing strongly penalizes too-short predictions.

PLAN (results/NB107_IOU_STUCK_PLAN.md): RESUME from ep18 (user: do NOT restart)
with the existing-but-OFF S1.4 lever `--lane-y-reweight near` ON, which
up-weights the near (bottom) rows where length is abandoned so shortening costs
more. Re-judge on curve-F1, not pixel-IoU. Optional B2 (length hinge) only if
length_mean still <~0.12 by ~ep28.

WIRED THIS CONVERSATION: `--resume` flag in train_lane_only.py (path or 'auto')
-> model.train(resume=...). Verified vs vendored trainer.check_resume: restores
optimizer/epoch/best_fitness/EMA, runs 19->120, and reloads the ckpt's saved
args (only imgsz/batch/device/epochs override) -- the lane levers ride env vars
so --lane-y-reweight DOES apply on resume. Unit-tested 4 branches; py_compile OK.
COLAB NOTE: /content/ is wiped between sessions -> next time restore last.pt from
Drive (training_runs/checkpoints/laneiou_both_full/last.pt) and pass that path
(or restore the run dir then --resume auto). last.pt is NOT in the local mirror.

NEXT: build NB108 (= NB107 recipe + `--resume ... --lane-y-reweight near`), run
~20-30 more epochs, confirm length_mean recovers + F1/curveIoU peak. CONDUCT
next conversation per user.

--- follow-up (same day): NB108 built + length-shrink fix UPGRADED ---
On reviewing the code (prompted by "did you do anything to fix the shrink?"),
found that --lane-y-reweight near is INDIRECT: it reweights the IoU loss on
x-offsets (masked by GT-target validity, lane_losses.py:50,61) and never touches
the `length` field (col 5) that is shrinking. length is supervised only by the
SYMMETRIC xytl smooth-L1 (1/4 of a mixed-scale mean) -> the model freely shortens
to dodge x-error. So added the DIRECT fix:
  - loss.py LANE_LEN_HINGE_W: xytl_loss_sum += w*((gt_len-pred_len).clamp(min=0)
    /n_strips).mean() on MATCHED priors only (asymmetric; symmetric L1 still caps
    over-long). Off by default. train_lane_only.py --lane-len-hinge flag.
  - Unit-tested (test_sprint1_levers test 5: short>0, exact=0, long=0, scales w).
    Suite 6/7 (baseline). No shim change (inline in MTDETRDLoss).
  - NB108 now runs --lane-len-hinge 2.0 + --lane-y-reweight near. 2.0 = starting
    estimate (tunable). CAVEAT: length_mean (val.py:509) averages ALL 192 priors
    not just matched -> noisy proxy; watch pixel-IoU(lane)+mask_frac RISING and
    curve-F1/curveIoU holding instead.
