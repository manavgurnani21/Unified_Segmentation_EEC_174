# RMT-PPAD → CLRHead lane-only migration (Path-3)

This folder contains the phase-by-phase execution of the full Option A plan
from `external_repos/RMT-PPAD-main/.claude/worktrees/stupefied-torvalds-a6544b/yolop_vehicle_lane/stage2/vendor/RMT-PPAD/tutorial/appendix-path3-implementation-prompt.md`.

**Goal:** produce a model = RMT-PPAD backbone + GCA + MTDETRDecoder (detection)
but with RMT-PPAD's lane mask decoder REPLACED by CLRKDNet's curve-based CLRHead.
Drivable area is removed from the new model entirely.

**Why:** RMT-PPAD's published BDD100K lane IoU is 56.8 %; CLRKDNet's CULane
F1 is 80.71 %. The hypothesis is that the curve-regression head produces
cleaner lane representations than per-pixel mask + post-processing — and that
GCA + MTDETRDecoder's detection branch is unaffected.

---

## Phase status

| Phase | Task ID | Status | Days est | Acceptance criterion |
|---|---|---|---|---|
| **P0** Baseline | #1 | ✅ **PASS** 2026-05-25 | 1-2 | All 4 reference numbers match exactly: mAP50=0.849, drivable_miou=0.926, lane_iou=0.568, lane_acc=0.847. `results/baseline_metrics.json` finalized. |
| **P1** Data conversion | #2 | ✅ **PASS** 2026-05-25 | 3-5 | 70k train + 10k val .pt files (slot-aligned (8,78) tensors); verifier 10/10 IoU ≥ 0.30; tarball at `datasets/lane_targets_clr_v1_polyline.tar.gz`; empty rate 7.4%/7.2% |
| **P2** Adapter | #3 | ✅ **PASS** 2026-05-25 | 1 | Shape unit test passes; 49,536 params; gradients flow through all 3 levels |
| **P3** Square CLRHead | #4 | ✅ **PASS** 2026-05-25 | 2-3 | 192 priors render as 32-left + 128-bottom + 32-right; tensor shape (192, 78); fixed verifier coord-flip |
| **P4** LaneSegHead integration | #5 | ✅ **PASS** 2026-05-25 | 2-3 | 35.3M params; `seg_head=LaneSegHead`, `adapter=GCAtoCLRAdapter`, `lane_head=CLRHeadForSquareImage`; no drivable in 941 named modules; eval forward returns `{lane_output: (1,192,78)}` |
| **P5** Loss redesign | #6 | ✅ **PASS** 2026-05-25 | 2-3 | `criterion.lane_only_mode=True`; `da_seg=0` exactly; all 4 lane keys finite (cls=16.60, xytl=4.96, iou=2.00, seg_aux=0.94); backward 720/780 params |
| **P6** Dataset/collate | #7 | ✅ **PASS** 2026-05-25 | 2-4 | DataLoader returns `img (B,3,640,640)`, `lane_targets (B,8,78)`, `lane_seg_mask (B,1,640,640)`; no drivable; MTDETRDataset path verified end-to-end |
| **P7** Validator/metrics | #8 | ✅ **PASS** 2026-05-25 | 2-4 | `MTDETRValidator.postprocess` dispatches on dict seg; lane_rasterize produces `(B,1,640,640)` mask via top-k+slot-aligned polylines; 13,914 lane pixels on synthetic boosted priors |
| **P8** Train + ablate | #9 | 🟡 in_progress | 7-14 | 4-row ablation table with at least 4 configurations |

**Total**: ~22-39 days of focused engineering.

---

## Folder layout

```
rmt_ppad_migration/
├── README.md                          (this file; lives outside notebooks for git visibility)
├── P0_baseline/                       Scripts to download checkpoint + run test.py
├── P1_data_conversion/                tools/lane_conversion/ from the plan
├── P2_adapter/                        GCAtoCLRAdapter source
├── P3_square_clrhead/                 CLRHeadForSquareImage source
├── P4_lane_head_integration/          LaneSegHead source + diffs to MTDETRDecoder
├── P5_loss/                           lane_losses.py + diffs to MTDETRDLoss
├── P6_dataset/                        diffs to dataset.py / collate_fn
├── P7_validator/                      lane_rasterize + lane_metrics + val.py diffs
├── P8_train/                          train configs, ablation runner, results
├── notebooks/                         NB79-NB87, one per phase, Colab-runnable
├── results/                           baseline_metrics.json, per-ablation jsons
└── vendor/                            shallow copy of RMT-PPAD framework (modifiable)
```

---

## Hard constraints from the source plan (don't violate these)

- **Do NOT modify** `external_repos/CLRKDNet-master/` or `external_repos/RMT-PPAD-main/`. Vendor first, then modify.
- **Do NOT delete** the original `TransformerSegmentationDecoder`. The old model stays usable for A/B comparison.
- **Do NOT change** the detection branch (`MTDETRDecoder` itself). Detection stays 256 channels, deformable attention, 300 queries.
- **Do NOT skip P1.** CLRHead cannot train on mask labels.
- **Do NOT add drivable back.** New model is explicitly drivable-free.
- **Do NOT bypass GCA by default.** Only bypass it as an ablation row in P8.

## Drive vs local Colab filesystem rule

**Tens of thousands of small files MUST be extracted into `/content/` (Colab's local SSD), NEVER into `/content/drive/MyDrive/...`.**

Drive's API throttles per-file, not per-byte. Empirical numbers from this project:
- One 1 GB .pt checkpoint -> Drive: ~30 s
- One 1 GB tarball expanded to 70 000 JPGs -> Drive: 2-3 hours, frequently fails
- Same 70 000 JPGs to `/content/`: 5-10 min, never fails

What goes where:
- **Drive**: source tarballs/zips, pretrained .pt checkpoints, final results (.json/.log/.png).
- **`/content/`**: extracted image/label/mask trees, working dirs for training, intermediate caches.

Anytime a phase script extracts a multi-file archive, the destination must start with `/content/`. The data flow is: Drive blob -> local extraction -> training reads from local -> training writes outputs back to Drive.

---

## Migration log policy

Every phase writes a `migration_log.md` under its phase folder when something
unexpected happens (compatibility issue, bug, design decision). The next AI
context window reads this log on entry. Don't silently retry — document.

---

## Current pointer

See `P0_baseline/` for the next concrete action.
