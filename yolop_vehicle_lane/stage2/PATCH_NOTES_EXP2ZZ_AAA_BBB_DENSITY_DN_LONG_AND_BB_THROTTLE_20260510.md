# Stage 2 Exp2ZZ + Exp2AAA + Exp2BBB — tune density, let DN queries converge, throttle backbone

## Reading NB52 / NB53 / NB54 — the three sharper findings

### NB52 (Exp2WW — anchor + topk_fixed K=8 + VFL)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | pos-neg gap | val_det |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.178 | 0.242 | 0.013 | 0.000 | 0.259 | 0.001 | 2.13 |
| 10 | 0.247 | 0.382 | 0.035 | 0.000 | 0.276 | 0.001 | 1.96 |
| 20 | **0.256** | 0.355 | 0.043 | 0.070 | **0.279** | **0.003** | 1.79 |

**Geometry crashed from NB48's 0.544 to 0.256.** K=8 spreads geometric gradient across 8 priors per GT — each gets 1/8 of the per-GT pull, so none locks onto the true curve. The pos-neg gap doubled from 0.001 to 0.003 — directionally right but K is wrong.

### NB53 (Exp2XX — query head K=64 + DAB + DN + VFL)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | pos-neg gap | val_det |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.192 | 0.118 | 0.021 | 0.283 | 0.435 | **0.177** | 2.12 |
| 7  | 0.152 | 0.112 | 0.021 | 0.456 | **0.476** | 0.157 | 3.10 |
| 20 | 0.195 | 0.144 | 0.018 | **0.374** | 0.374 | 0.062 | 3.35 |

**val_lane_best_f1=0.476 at epoch 7 — the highest cls discrimination ever measured on the project.** DAB-anchor positional encoding + DN-DETR denoising queries definitively cracked the cls representation problem. But:
- matched_iou stayed weak (0.195) — queries didn't have enough epochs to ground geometrically.
- val_det collapsed from 2.12 → 3.35 between epoch 7 and 8 — joint conflict kicked in once the backbone unfroze.

### NB54 (Exp2YY — anchor + topk_fixed + VFL + 70K + det rescue)

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap | val_det | val_map50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.234 | 0.366 | 0.016 | 0.051 | 0.274 | -0.001 | 2.66 | 0.0009 |
| 6  | **0.289** | **0.420** | **0.062** ✓ | **0.237** | 0.299 | 0.020 | 3.10 | 0.000 |

**decoded_f1 = 0.062 — new project record** (vs NB48's 0.050, NB40's 0.043). val_lane_f1 = 0.237 — second-highest ever. But:
- matched_iou = 0.289 (vs NB48's 0.544 — half lost, same K=8 dilution problem)
- val_det = 3.10, val_map50 = 0 — det broken at full data despite `lambda_det=3.0`.

## Diagnosis after 23 experiments

The three problems are now **separated and partially solved**:

| problem | who solves it | what NOT to do |
|---|---|---|
| **Geometry** | anchor head 192 + dynamic-k (NB48, matched_iou=0.544) | topk_fixed K=8 (NB52/54 drop to ~0.26-0.29) |
| **Cls discrimination** | DAB anchors + DN denoising + VFL + Hungarian (NB53, val_lane_best_f1=0.476) | matched_existence on 192-anchor (gap stuck at 0.01) |
| **Joint conflict at full data** | **unsolved** — `lambda_det` re-weighting from 1→2→3 across NB48/51/54 didn't help | rely on Kendall uncertainty (it drifts toward lane) |

The joint conflict in particular has a sharper diagnosis now: **re-weighting head losses doesn't matter because both gradients backpropagate through the same backbone tensor, and at full data scale the lane gradient norm dominates the backbone updates regardless of head-level scaling.** The fix needs to be at the BACKBONE update path, not the head loss path.

## Three new experiments

### Exp2ZZ (NB55, exp50) — anchor + topk_fixed K=3 + VFL (3K)

**Density tuning.** K=8 was too many positives per GT (geometry crashed). K=3 = ~15 positives/image, close to dynamic-k's empirical 8-12.

Single config diff vs NB52: `topk_fixed_per_gt: 8 → 3`. Tests whether the stability-vs-dilution sweet spot exists at K=3.

### Exp2AAA (NB56, exp51) — query + DAB + DN + VFL + 30 epochs (3K)

**Let DN queries converge.** NB53's val_lane_best_f1 peaked at 0.476 at epoch 7 then degraded. DN-DETR papers note queries need 50+ epochs at standard batch sizes to fully ground — 20 was premature.

Diffs vs NB53:
- `end_epoch: 20 → 30`
- `lambda_det: 1.0 → 1.5`, `use_uncertainty: true → false` (prevent the late-training det collapse NB53 showed)
- `warmup_epochs: 2 → 3`

### Exp2BBB (NB57, exp52) — anchor + VFL + full 70K + backbone-LR throttle

**The joint-conflict fix.** Re-weighting head losses (NB48: 1.0, NB51: 2.0, NB54: 3.0) didn't help because both gradients backprop through the shared backbone. Throttle the BACKBONE updates instead:

- `backbone_lr_mult: 0.1 → 0.01` (the only critical change)
- `lambda_det: 3.0 → 1.5` (gentler — expecting backbone fix to fix det)
- `lambda_lane: 0.5 → 1.0` (don't suppress lane unnecessarily)
- matcher back to `dynamic_k` (NB48's geometry champion)

With backbone LR = 2e-6 (10x slower than default), the backbone moves very slowly under both task gradients. Both heads adapt to the slowly-changing backbone rather than fighting over it. This is the standard DETR recipe (`lr_backbone=1e-6` while `lr_main=1e-4`).

## Code state

- No new code changes (existing infrastructure is enough — `lane_assigner='topk_fixed'`, `LaneQueryHeadAnchorDN`, `backbone_lr_mult` were already in the codebase).
- 3 new configs (`exp50`/`exp51`/`exp52`), 3 new notebooks (NB55/NB56/NB57), NB08 patched.
- `python -m compileall stage2/{fusion,metrics,scripts}` passes.

## Pass criteria

### Exp2ZZ at epoch 20
- `val/matched_line_iou ≥ 0.40` (recover most of NB48's 0.544 with stable labels)
- `pos_score − neg_score ≥ 0.02` (≥ 7× NB52's 0.003)
- `val/lane/decoded_f1 ≥ 0.07` (beat NB54's 0.062)

### Exp2AAA at epoch 30
- `val/matched_line_iou ≥ 0.30` (1.5× NB53 with more epochs to ground queries)
- `val/lane_best_f1 ≥ 0.40` (maintain NB53 peak)
- `val/lane/decoded_f1 ≥ 0.05` (2× NB53)
- `val_det ≤ 2.5` at epoch 30 (det rescue holds)

### Exp2BBB at epoch 6 — the smoking gun
- **`val_det ≤ 2.5` AND `val/det/map50 ≥ 0.005`** — if backbone throttling fixes det at full data, this hits.
- `val/matched_line_iou ≥ 0.45` (small acceptable regression from NB48's 0.544)
- `train/grad_cosine_epoch_mean ≥ 0` across most epochs.

## Run order

1. **Exp2ZZ first** (NB55, ~30 min, 3K). Cheapest signal: does K=3 recover NB48-level geometry?
2. **Exp2AAA** (NB56, ~50 min, 3K). DN queries with more epochs.
3. **Exp2BBB** (NB57, ~60-80 min, full data). Test the backbone-throttle hypothesis on the det collapse.

Independent of each other and all prior NBs.

## Why I chose these over the user's CVPR suggestions

The user's hints (GELAN-PGI, AdaBins, NaviBridger, STAL, PCGrad, DDAD) remain queued. My empirical analysis says:

- The K=8 dilution finding is concrete and the K=3 single-knob test is the lowest-cost way to validate the topk_fixed direction. If K=3 recovers geometry AND keeps cls gap > 0.02, the matcher fix becomes a real win.
- DN-DETR's 7-epoch peak in NB53 says "the architecture works but training was too short" — the obvious next test is more epochs.
- The joint conflict at full data is a BACKBONE-update problem, not a head-weighting problem. Throttling `backbone_lr_mult` is the standard DETR recipe and a 1-line config change. PCGrad would be the next escalation if Exp2BBB also fails.

If Exp2BBB rescues det at full data AND Exp2ZZ recovers geometry, combining them (full data + K=3 topk_fixed + VFL + backbone_lr_mult=0.01) would be the next Exp2CCC and likely the final ship-able model.
