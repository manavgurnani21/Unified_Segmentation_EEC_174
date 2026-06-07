# Plan after inspecting NB88 / NB89 / NB92 training runs

Date: 2026-05-29. Author: analysis of `training_runs/checkpoints/*` +
`training_runs/notebook_logs/exp01-71` + prior research patch notes.

---

## 1. What the runs actually show

Parsed `results.csv` + `log_epoch_*.json` for all five completed runs.

| Run | NB | epochs | det mAP50 (peak → end) | **IoU(lane)** | subacc(lane) |
|---|---|---|---|---|---|
| clr_lane_default        | 88 | 49 | 0.825 @ep15 → 0.798 | **0.0718 FROZEN** | 0.5000 flat |
| clr_lane_no_square_priors | 89 | 48 | 0.824 @ep15 → 0.795 | **0.0718 FROZEN** | 0.5000 flat |
| polyline_baseline_small | 92 | 30 | 0.699 @ep15 → 0.672 | **0.0710 FROZEN** | 0.5000 flat |
| bezier_cubic_no_lcm     | 92 | 30 | 0.696 @ep15 → 0.666 | **~0.001 (noise→0)** | 0.4998 flat |
| bezier_lcm_gamma001     | 92 | 30 | 0.698 @ep15 → 0.668 | **0.00→0.07→0.003 (unstable)** | 0.495 |

Two systemic failures, both independent of bezier-vs-polyline:

### 1a. Lane IoU never learns (THE blocker)
- Polyline runs: IoU is **byte-constant** at 0.0718/0.0710 for 30-49 epochs
  to 4 decimal places. subacc is **exactly 0.5000** (random classifier).
- Meanwhile training `ll_seg` DOES decrease (1.305 → 0.593 on NB88). So
  the lane LOSS goes down but the decoded/rasterized lane IoU is flat.
- Bezier runs: IoU is noisy and collapses toward ~0.003. The bezier
  curves move but converge to garbage.

**The lane head is not producing improving curve predictions that the
validator's rasterizer can measure.** The `ll_seg` decrease comes almost
entirely from CLRHead's auxiliary seg-decoder convnet (a separate
per-pixel head), NOT from the prior-curve regression that feeds IoU.

### 1b. Detection mAP peaks ~epoch 15 then declines
- Every run: mAP50 climbs to ~0.82 (NB88/89) / ~0.70 (NB92 10k) by
  epoch 15, then **monotonically declines** to end.
- `best_fitness` locks at epoch 13-19 and is never beaten. A 250-epoch
  run would waste ~230 epochs and end WORSE than epoch 15.

---

## 2. The decisive context: this problem was already studied ~70 times

`training_runs/notebook_logs/` contains `exp01`-`exp71` and the prior
research worktree
(`external_repos/RMT-PPAD-main/.claude/worktrees/stupefied-torvalds-a6544b/
yolop_vehicle_lane/stage2/PATCH_NOTES_EXP2*.md`) documents Exp2G → Exp2VV:
a ~2-week campaign attacking THIS EXACT "lane cls collapse / frozen IoU"
problem. The conclusions (verbatim from the patch notes + exp logs):

### Root cause (proven): dynamic-k matching causes cls collapse
> "The cls collapse on the anchor head wasn't a loss-function bug — it's
> the matching scheme. dynamic-k labels the same prior as positive in
> some batches and negative in others, so the cls converges to a uniform
> sigmoid as the only stable equilibrium."

A prior is matched as POSITIVE in some batches, NEGATIVE in others →
the binary classifier's only stable fixed point is "output 0.5 for
everything" → all priors pass conf_threshold equally → rasterizer draws
the same anchor fan every epoch → **IoU frozen**. This is exactly the
0.0718-forever signature in NB88/89/92.

### What was tried and FAILED to fix cls collapse (gap stayed ≤ 0.015)
- ASL cls rescue (Exp2H), OHEM (Exp2I), separate cls path (Exp2J)
- IoU regression / QFL / sqrt-target (Exp2K/L/M)
- **VFL / varifocal loss alone (Exp2QQ/NB46)** — "the VFL equilibrium
  just shifted to a different uniform value"
- dense mask supervision (Exp2W/X), dual-score (Exp2LL/MM),
  lambda oscillation fixes (Exp2Y/Z/AA)

### What PARTIALLY worked
- **K=64 query head + Hungarian 1-to-1 matching (Exp2RR/NB47)**: FIRST to
  break collapse. pos-neg gap 0.099 (10× historical), val_lane_f1 0.246,
  best_f1 0.344. BUT geometry weak (matched_iou 0.27 vs anchor's 0.54).
- **Anchor head geometry is excellent** (matched_iou up to 0.544) but cls
  always collapses on it.
- The two are **orthogonal**: Hybrid prior-query head (Exp2TT) was the
  proposed combination (stage-1 anchors for geometry + stage-2 K=12
  queries with Hungarian for cls).
- CULane pretrain/KD (exp69/70), cls-separation + VFL + full data
  (exp59-71): geometry matched_iou 0.55, but **decoded_f1 stuck at ~0.05**
  and **detection map50 collapsed to ~1e-5 under hard full-data lane
  training** (the joint conflict).

### Honest status of prior research
After ~70 experiments the team reached: anchor geometry 0.54, but decoded
lane F1 never exceeded ~0.05-0.07 with detection preserved. The cls
collapse + joint det/lane conflict were **never fully solved** on the
anchor head. The K=64 query head broke cls (f1 0.25) but never got
geometry + detection + scale together.

---

## 3. What this means for the current migration

My clean "Option A" migration (P0-P8 + bezier_lcm) **reproduced the known
failure** because it is built on the same substrate:
- P5 `lane_losses.assign` = `dynamic_k_assign` (confirmed)
- B5 `bezier_losses.assign_bezier` = `dynamic_k_assign_bezier` (confirmed)

Both use dynamic-k → both collapse cls → frozen/zero decoded IoU. The
bezier representation changes the curve PARAMETERIZATION but inherits the
MATCHING scheme, so it cannot fix the blocker.

**Consequence for NB92's brief ablation: it is uninformative.** All three
rows (polyline / bezier / bezier+LCM) sit on collapsed cls. They differ
only in detection-mAP noise and in how their already-broken lane IoU
wobbles near zero. The aggregator's "winner = polyline (IoU 0.071)" is
just the frozen-anchor baseline — not a real result. **Do not promote any
NB92 row to a 250-epoch NB93 full run.** That would burn ~30 GPU-hours to
reproduce a known dead end.

An accidental upside in the current runs: because the lane weights were
cut 5× (joint-stability work), the lane head is so inert that DETECTION
trains fine (map50 0.82). The prior research's det-collapse happened when
lane was trained hard. So we sit on the opposite side of the det/lane
frontier — detection healthy, lane dead.

---

## 4. The plan going forward

### Phase 0 — STOP and re-scope (this document)
- Do not launch NB93 full training of any NB92 row.
- Treat NB88/89/92 as **substrate-validation runs**: they prove the
  pipeline trains end-to-end, saves per-epoch, syncs to Drive, and
  reports the full metrics table. That infrastructure WORKS. The science
  (lane learning) does not yet.

### Phase 1 — Make decoded lane IoU move at all (highest leverage)
Adopt the single proven fix from the prior research: **replace dynamic-k
matching with Hungarian 1-to-1 matching** in the lane loss. This is what
broke the cls collapse in Exp2RR/NB47 (gap 0.01 → 0.099, f1 0 → 0.25).

Concretely, on the EXISTING anchor head (cheapest path):
1. Add a `match='hungarian'` option to `P5_loss/tools/lane_losses.py`
   `assign()` using `scipy.optimize.linear_sum_assignment` on the same
   cost matrix (focal + distance + LineIoU). One positive prior per GT.
2. Mirror it in `B5_loss/tools/bezier_losses.py` `assign_bezier()`.
3. Add `--lane-match {dynamic_k,hungarian}` to both train scripts,
   default `hungarian`.
4. Re-run ONE short probe (30 epochs, 10k subset) on the polyline head.
   PASS = decoded IoU(lane) climbs past 0.10 and rises across epochs
   (not frozen). This is the go/no-go gate for everything downstream.

If Hungarian-on-anchor still collapses geometry-vs-cls (prior research
suggests anchor + Hungarian may undersupply positives), fall back to the
**K=64 query head** path (Exp2RR), which the prior research proved breaks
cls — port `LaneQueryHead` from the worktree rather than rebuild.

### Phase 2 — Re-establish the det/lane balance
Once lane IoU is genuinely learning, detection WILL start to feel the
joint conflict (prior research: full-data lane gradient starves det).
Apply the proven counter-measures BEFORE they bite:
- Fixed loss weighting with explicit det boost (`lambda_det` up,
  `lambda_lane` down), NOT Kendall uncertainty (Exp2VV finding:
  uncertainty migrates weight away from det at scale).
- Keep the 5× lane-weight reduction as the STARTING point — we are
  already in the det-safe regime; raise lane weight gradually only as
  far as decoded IoU keeps climbing without det mAP collapsing.

### Phase 3 — Fix the detection peak-then-decline
Independent of lane. mAP peaks ~ep15 then declines in every run.
- Add early-stopping on detection fitness (patience ~15) OR shorten the
  schedule, so we keep the ep15 weights instead of ending worse.
- Investigate whether the decline is cosine-LR overshoot or slow lane
  corruption of the shared backbone (compare a det-only control run).

### Phase 4 — ONLY THEN ablate bezier vs polyline vs LCM
The bezier/LCM ablation is scientifically meaningful **only on a
substrate where lane IoU actually learns**. Re-run NB92's three rows on
the Phase-1 (Hungarian) substrate. Now the question "does bezier's
parameterization help over polyline" has a measurable answer because both
heads will produce non-frozen IoU.

### Phase 5 — Full training of the real winner (future NB93)
Promote the Phase-4 winner to 250 epochs / 70k, with Phase-2 det balance
and Phase-3 early stopping in place.

---

## 5. Concrete next actions (small, ordered, each independently shippable)

1. **[infra, done-ish]** Keep all the NB88/89/92 bugfixes already shipped
   (val_period=1, picklable cfg.haskey, bezier rasterizer dispatch,
   autograd cat-fix, per-epoch Drive sync, metrics table). These are
   correct and reusable.
2. **[Phase 1]** Implement Hungarian matching option in both loss files +
   `--lane-match` CLI flag. ~60 LOC. (next code change)
3. **[Phase 1 gate]** New probe notebook NB94: polyline head, Hungarian
   matching, 30 ep / 10k. Watch decoded IoU(lane). Go/no-go.
4. **[Phase 1 fallback]** If anchor+Hungarian insufficient, port
   `LaneQueryHead` (K=64, Hungarian, VFL) from the worktree.
5. **[Phase 3, parallel]** Add det early-stopping (patience=15) to both
   train scripts so no run ends worse than its peak.

## 6. What NOT to do (lessons bought with ~70 prior experiments)
- Do NOT try to fix cls collapse with another loss tweak (ASL, OHEM,
  QFL, plain VFL, dual-score) — all failed; it's the matching, not the
  loss.
- Do NOT use Kendall uncertainty weighting for det/lane balance at full
  data — it accelerates det collapse.
- Do NOT promote a dynamic-k run to full 250-epoch training.
- Do NOT read NB92's ablation winner as a real result.
