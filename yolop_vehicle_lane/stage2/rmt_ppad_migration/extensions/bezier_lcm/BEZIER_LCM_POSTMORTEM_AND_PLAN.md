# Bezier + LCM — NB92 post-mortem, root cause, and improvement plan

Date 2026-05-30. Supersedes the *results* sections of `PLAN.md` (which is
the original pre-implementation design doc). This document answers four
questions the way the runs actually came out:

1. How did Bezier and LCM perform?
2. Why? (root cause)
3. How do we improve them? (what is already fixed + what is left)
4. What additional ablations/experiments do we need, in what order?

---

## 1. How did Bezier and LCM perform? (NB92 brief ablation, 10k subset, 30 ep)

Measured from the NB92 run folders (`bezier_cubic_no_lcm`,
`bezier_lcm_gamma001`) and their `full_train.log`s:

| Row | Lane IoU (start → peak → end) | Det loss | ll_seg loss | Verdict |
|---|---|---|---|---|
| `polyline_baseline_small` (ref) | climbs to ~0.07–0.09* | falls | falls | works |
| `bezier_cubic_no_lcm` | 0.0006 → **0.0238** (ep10) → 0.0211 | 26 → 10 | 0.8 → 0.54 | **learns but plateaus ~4× below polyline** |
| `bezier_lcm_gamma001` | 0.0000 → **0.0118** → 0.0123 | falls | falls | **LCM roughly HALVES bezier IoU — it HURTS** |

\* polyline reference: the full-data NB97 polyline reached IoU 0.089
(pixel-IoU, which saturates ~0.09 for thin lines). The 10k/30ep
`polyline_baseline_small` is lower but still clearly above both bezier
rows.

Two hard facts:
- **Bezier underperforms polyline by ~4×** on the same budget.
- **LCM makes bezier worse, not better** (0.012 vs 0.024).

The bezier loss *does* go down (det 26→10, ll_seg 0.8→0.54), so the model
is training — it is the *rendered mask overlap* that stays tiny
(`mask_frac` ~0.06 vs polyline's higher fraction). The curve is being fit
in control-point space but barely shows up on the rasterized mask.

> Caveat that matters: these NB92 bezier numbers predate (a) the
> `lane_rasterize.py` length-scaling fix that unfroze the polyline IoU,
> and (b) all the `[lane-geom]`/`[lane-mask]` diagnostics. The bezier path
> uses its OWN rasterizer (`_bezier_lanes_to_mask`), so the length fix did
> not touch it — but it also means the bezier rows were never observed
> with the instrumentation that diagnosed the polyline. That gap is now
> closed (sec 3).

---

## 2. Root cause — the "5% stub" t-range init bug (high confidence)

The bezier head emits, per prior, a validity range `[t_start, t_end]` and
only the curve segment in that range is rendered. The head's
`reg_layers` was initialized with **zero bias** and tiny weights
(`std=1e-3`), so at initialization, for every prior:

```
t_start = sigmoid(reg[8]) = sigmoid(0)        = 0.50
t_end   = max(sigmoid(reg[9]), t_start+0.05)  = 0.55
span    = t_end - t_start                      = 0.05   ← only 5% of the curve
```

So **at step 0 each Bezier renders only the middle 5% of its span** — a
short stub floating in the centre of where the lane should be. The
polyline head, by contrast, draws the *full* length from step 0 (its 72
x-offsets all render).

Why the stub is self-trapping:
- The training IoU loss (`bezier_liou_loss`) renders the pred curve over
  `[t_start, t_end]` and compares it **aligned** (sample index k vs k)
  against the GT curve spanning `[0, 1]`. With pred at `t∈[0.50, 0.55]`,
  pred sample k sits at a totally different y than GT sample k →
  intersection ≈ 0 → IoU ≈ 0 → loss saturates at 1.0 → **almost no
  gradient flows back to widen the range.**
- The only thing pulling the range open is the weak `val_loss` term
  (smooth-L1 of `t_start,t_end` vs the target's `0,1`), which moves it
  slowly over many epochs. That is exactly the "learns but plateaus low"
  signature we see.

This is the dominant cause. Two secondary factors compound it:

- **Control-point bottleneck.** 4 control points must describe the whole
  lane vs the polyline's 72 independent per-strip offsets. Lower capacity
  and a harder credit-assignment path (a single P1/P2 move bends the
  entire curve). This caps the achievable IoU even once the range is open
  — but it is a *ceiling* effect, not the reason for the 4× gap.
- **No augmentation + predates length fix.** The bezier trainer runs with
  `fliplr=0.0` and the rows were measured before the diagnostics existed,
  so they were never tuned.

### Why LCM hurts (consistent with the same root cause)
LCM blends K=1 (straight chord P0→P3), K=2 (quadratic), and K=3 (cubic)
using the SAME control points. When the cubic itself is a barely-visible
5% stub, the mixture mostly averages toward the straight chord, which
fits the (already tiny) curve even *worse*, and the Occam penalty
(γ=0.01) actively pushes weight toward the simpler, worse-fitting degrees.
**LCM cannot help until the base bezier actually draws full curves.** Its
NB92 result is not evidence against LCM as an idea — it is evidence that
LCM was stacked on a broken base.

---

## 3. What is ALREADY fixed in this commit

1. **t-range bias init** (`B4_head/tools/lane_bezier_head.py`). After the
   `reg_layers` init, slot-8 bias set to **−4** (→ `t_start≈0.018`) and
   slot-9 bias to **+4** (→ `t_end≈0.982`). Every Bezier now spans ~96% of
   its curve from step 0, so the IoU loss is informative immediately and
   the head learns to *shape* a full-length curve instead of having to
   first discover it should grow one. This is consistent with the loss's
   own targets (`target t_start=0, t_end=1`), so the `val_loss` term
   reinforces rather than fights the new init.

2. **Bezier t-range diagnostics** (`vendor/.../mtdetr/val.py`). New
   per-epoch log line and metric:
   ```
   [bezier-trange] t_start mean=0.02 t_end mean=0.98 span=0.96
                   (span ~0.96 => full curve drawn; ~0.05 => 5% stub bug)
   metrics/lane_bezier_span(lane)
   ```
   This makes the fix *observable*: if `span` reads ~0.96 from epoch 1 the
   init took; if it collapses back toward ~0.05 the head is re-closing the
   range and we have a deeper problem.

Both auto-apply on the next NB92 re-run — NB92 imports `LaneBezierHead`
and the vendored `val.py`, so **no notebook rebuild is needed, just a
re-run** (after the user re-syncs Drive).

> **VERIFIED (2026-05-30 re-run, `bezier_cubic_no_lcm`, 30 ep):**
> `lane_bezier_span = 0.975` from **epoch 1** (was the 0.05 stub) and lane
> IoU **~doubled** (old peak 0.024 → 0.049). Phase 1 PASSED — the stub bug
> was the dominant cause. Remaining gap to polyline (~0.07) is the
> control-point bottleneck + aligned-IoU loss → next steps I1 (curve-
> sampling loss) + I2 (flip aug). `mask_frac` stays low (~0.027), confirming
> the curve draws full-length but still overlaps GT loosely.

---

## 4. How to improve Bezier and LCM (concrete, ranked)

### Already done (sec 3): t-range init + diagnostics. Highest leverage.

### Next, in priority order:

**I1. Curve-sampling IoU loss instead of aligned-index IoU.** The current
`bezier_liou_loss` is aligned (sample k vs k). Even with a full span,
aligned comparison is brittle when the y-parameterization differs. Switch
the eval/loss to compare pred vs GT **at matched y-rows** (resample both
curves at a common set of y values, compare x). This is how CLRKDNet's
LineIoU works for polylines and removes the parameterization mismatch.

**I2. Bezier-aware horizontal-flip augmentation.** The polyline path got
`fliplr=0.5` with synced lane-target flipping; the bezier path is still
`fliplr=0.0`. Flipping a bezier target is `P_x → (W−1)−P_x` for all 4
control-point x's (y's unchanged, t-range unchanged). Add a 16-D branch to
`augment.py`'s RandomFlip and turn on `fliplr=0.5` in
`train_bezier_brief.py`. **Verify visually first** (same rule as the
polyline flip: draw the flipped control points + rendered curve on the
flipped image and confirm alignment) before spending GPU hours.

**I3. Light curvature/anti-collapse term.** Risk register item: control
points can collapse toward the midpoint (always-straight). Add a small
reward for |P1−chord| + |P2−chord| (opposite-sign smoothness) so the head
is not biased to straight lines, OR simply rely on I1 first and only add
this if curves come out straight.

**I4. Re-test LCM only AFTER I1–I2 land.** With a working base bezier,
re-run `bezier_lcm_gamma001` and a `γ=0.001` variant. If LCM still hurts
with a healthy base, it is genuinely not worth it on BDD and we drop it.

**I5. Bezier curve-F1. [DONE this commit]** The curve-F1 metric now has a
16-D decode path (`_decode_lane_bezier` + a width-dispatching `_decode_row`
in `lane_curve_f1.py`), and `val.py`'s curve-F1 accumulator accepts
`shape[-1] in (78, 16)`. So bezier and polyline are now scored on the
*same un-saturated* yardstick (pixel-IoU saturates ~0.09 and can't tell
them apart). Decode math hand-verified: a full-span GT lane covers
y∈[11,148]; a 5%-stub covers only y∈[72,80] → IoU≪0.5 → correctly
unmatched (the metric reflects the bug, as it must).

---

## 5. Additional ablations / experiments needed

Ordered cheap→targeted. Each has a clear PASS gate so we do not burn GPU
on inconclusive rows.

| # | Experiment | Isolates | PASS gate |
|---|---|---|---|
| **A** | bezier_cubic re-run WITH t-range fix vs the old NB92 number | the init fix alone | `[bezier-trange] span ≥ 0.9` from ep1 AND IoU peak > 0.04 (≈2× the old 0.024) |
| **B** | + curve-sampling IoU loss (I1) | parameterization mismatch | IoU peak rises again vs A; loss no longer saturates at 1.0 early |
| **C** | + bezier flip aug (I2) | overfitting / data | val IoU peak-then-decline flattens (same benefit seen for polyline) |
| **D** | LCM on vs off ON THE FIXED BASE (γ=0.01 and γ=0.001) | does adaptive degree help once base works | LCM ≥ base IoU AND `complexity_score` meaningfully ≠ 3.0 (LCM actually used) |
| **E** | bezier vs polyline on **curve-F1** (I5), matched budget | which representation is really better | decisive ranking on the un-saturated metric |
| **F** | refine_layers 1 → 2/3 (capacity) | control-point bottleneck | IoU rises with stages; if flat, bottleneck is not the limiter |

Deferred until the above resolve (from `PLAN.md` sec 3.1, unchanged):
quintic K=6, LCM density factor, Gumbel-hard selection, K-supervision
from P1, LCM-at-all-stages, smoothness loss (subsumed by I3 if needed).

---

## 6. Concrete step-by-step plan (what to actually run, in order)

**Phase 0 — landed in this commit (no GPU):**
- [x] t-range bias init fix (`lane_bezier_head.py`).
- [x] `[bezier-trange]` diagnostic in `val.py` (span = t_end − t_start).
- [x] I5: bezier (16-D) decode path in curve-F1 + `val.py` accepts 16-D,
      so bezier and polyline rank on the same metric. Decode math
      hand-verified.
- [x] Compile-verified all touched files.

Pulled I5 forward into Phase 0 because it is pure-additive and
locally verifiable; I1/I2/I3 stay in Phase 2 because they change the
training loss/augmentation and should wait for the Phase-1 span
confirmation (and I2 needs visual verification of the flipped curves
before GPU time).

**Phase 1 — confirm the fix (Exp A), CHEAP, do first:**
1. User re-syncs Drive (so Colab picks up the head + val.py edits).
2. Re-run NB92 row 2 (`bezier_cubic_no_lcm`, `fresh=True`).
3. Read `[bezier-trange]` on epoch 1: **span must be ≈0.96.** If not, stop
   — the fix did not propagate (stale Drive) and nothing downstream is
   valid.
4. Read the IoU curve: PASS if peak > ~0.04 (≈2× the old 0.024). This
   confirms the stub bug was the dominant cause.

**Phase 2 — close the remaining gap (only if Phase 1 passes):**
5. Implement I1 (curve-sampling IoU loss) → re-run as Exp B.
6. Implement + **visually verify** I2 (bezier flip) → Exp C.
7. Add bezier curve-F1 (I5) so Exp E can rank bezier vs polyline fairly.

**Phase 3 — re-evaluate LCM (Exp D) on the fixed base:**
8. Re-run `bezier_lcm_gamma001` and a `γ=0.001` variant. Decide keep/drop
   LCM on the curve-F1 ranking, not pixel-IoU.

**Decision gate before any full (NB93) bezier training:**
- Bezier must beat polyline on **curve-F1** (Exp E) at the 10k/30ep
  budget, OR tie within noise with a clear memory/speed advantage, before
  we spend ~30 GPU-h on NB93. If bezier cannot beat polyline even with the
  stub bug fixed, we publish both numbers and keep polyline as the
  production head (consistent with `PLAN.md` sec 9).

---

## 7. What NOT to do
- Do not read the NB92 bezier numbers as the verdict on the
  representation — they were measured on the broken (5% stub) head.
- Do not judge bezier vs polyline on pixel-IoU (saturates ~0.09); use
  curve-F1.
- Do not re-enable LCM before the base bezier draws full curves (it will
  keep hurting and waste the ablation).
- Do not start NB93 full bezier training until the Phase-2 decision gate
  passes.
