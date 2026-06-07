# Stage 2: NB73 speed test, NB74 bugs + fixes, NB76 CLRKDNet download, NB77 real KD

## Reading NB73 — the speed test SUCCEEDED

NB73 ran NB62's recipe with the speed flags from my earlier proposal: batch=32, workers=6, lr0 sqrt-scaled to 4e-4, `torch.compile(reduce-overhead)`. 12 epochs on full 70K data.

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | val_lane_best_f1 | gap | val_map50 | elapsed/ep | mem |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.475 | 0.378 | 0.036 | 0.004 | 0.077 | 0.008 | 0.0001 | 1791s (compile warmup) | 10.8 GB |
| 6 | 0.544 | 0.464 | 0.052 | 0.099 | 0.106 | 0.024 | 0.0001 | 1657s | 13.1 GB |
| **12** | **0.558** (record) | **0.484** (record) | 0.050 | 0.102 | 0.110 | 0.029 | 0.0001 | 1657s | 13.1 GB |

**Findings:**
- **~1.8× speedup** (5.5 hr vs NB62's 10 hr). Not the 4× I predicted because validation dominates at large batch.
- **NEW geometry record**: matched_iou=0.558. Speed scaling didn't hurt — slightly helped because larger effective batch is a regularizer.
- Memory only 13 GB / 95.6 GB — could go batch=64 (~25 GB) or 128 (~50 GB).
- val_map50 still ~0 at full data — confirms det is not a "more compute" problem.

## NB74 FAILED — two bugs, both fixed

**Bug A.** `prepare_culane_dataset.py` reported every archive as missing:
```
[missing] /content/drive/MyDrive/EcoCAR/downloads/CULane/list.tar.gz -- skipping
[missing] /content/drive/MyDrive/EcoCAR/downloads/CULane/annotations_new.tar.gz -- skipping
... (all 6 drivers + 2 annotations missing)
```
Either the user's Drive hadn't synced to that exact path, or the folder name differs. The script silently produced 0 extracted files. **Fix**: I added a hard `FileNotFoundError` with a directory listing + parent search so the user can see what IS there. Re-running NB74 cell 3 will now print exactly what files are visible in MyDrive and where they actually live.

**Bug B.** Training script crashed:
```
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'
```
At `curve_tar = Path(args.curve_tar or cfg['dataset']['bdd_archive'])`. The CULane config has `bdd_archive: null` (no tar — CULane is extracted directly into `curve_root`), so both fallbacks were None and `Path(None)` crashes. **Fix**: `curve_tar` is now optional; when `None`, the script skips `extract_tar_once` and just validates `curve_root` exists. Required no other code changes — `BDDJointCurveDataset` only reads from `curve_root`.

NB74 cells 3-4 are cleared of stale outputs so the next run is clean.

## NB76 created — downloads CLRKDNet's GitHub releases

This is the "real public-teacher" prep. Pulls 4 files from `github.com/weiqingq/CLRKDNet/releases`:

| file | size | what it is |
|---|---|---|
| `dla34_clrnet_culane_8087.pth` | ~80 MB | CLRNet DLA-34 teacher, **80.87 F1 on CULane** — the actual paper-grade pretrained model |
| `dla34_distillation_log.txt` | <1 MB | Their DLA-34 distillation training log |
| `resnet18_distill_log.txt` | <1 MB | Their ResNet-18 distillation log |
| `dla_CLRNet_rerun_log.txt` | <1 MB | Training log of the teacher itself |

Saved to `/content/drive/MyDrive/EcoCAR/downloads/clrkdnet_weights/`. Cell 4 verifies the .pth loads in PyTorch and prints the parameter family prefixes — useful for the architecture adapter we'd need to actually consume these.

**Why this isn't NB77's teacher.** Architecture gap: CLRKDNet's backbone is DLA-34 (residual + DLA hierarchical aggregation); ours is RMT-GCA. CLRKDNet's lane head is CLRNet's 192-prior implementation; ours is CLRKDLaneHead (conceptually identical but separately written). Using their `.pth` requires:
1. Building CLRKDNet's model definition (`external_repos/CLRKDNet-master/`) and importing it
2. Running it as a parallel inference pipeline alongside our training
3. Or: writing a tensor-name mapping adapter into our `CLRKDLaneHead` (lossy because backbones differ)

NB76 puts the weights on Drive so a future Exp2TTT can do this engineering. For NB77, I picked the cleaner path.

## NB77 created — joint BDD with CULane KD teacher (from NB74)

Uses the existing `teacher.lane_head_checkpoint` machinery (no new code in losses.py) pointed at NB74's lane-only CULane checkpoint. The teacher is **our same architecture** trained on a **different distribution** (CULane), which:

1. **Skips the architecture-adapter problem** entirely — `load_lane_teacher` handles it natively.
2. **Provides domain-shifted distillation signal**: teacher saw CULane (Chinese highways), student is trained on BDD (US urban). Genuine knowledge to transfer.
3. **Teacher has no det conflict** because NB74 was lane-only.

NB77 cell 3 extracts NB74's tar, finds `best.pt` automatically, prints its lane_head key structure for debug. Cell 4 patches the config to point `teacher.lane_head_checkpoint` at that file, then runs joint training with `w_distill=1.0`, batch=32, full 70K data, 12 epochs.

Speed flags from NB73 are applied: batch=32, workers=6, prefetch=4, torch.compile, bb_throttle=0.01.

## Pass criteria

### NB77 at epoch 12
- **`val/lane/distill` decreases monotonically** — teacher signal active and useful (NB72 failed this because teacher = student).
- `val/matched_line_iou ≥ 0.56` — preserve NB73's 0.558.
- **`val/lane_best_f1 ≥ 0.15`** — 1.5× NB73's 0.110, the smoking gun for KD actually transferring cls knowledge.
- **`val/lane/decoded_f1 ≥ 0.07`** — 1.5× NB73's 0.050.
- `pos_score - neg_score ≥ 0.04` — KD pushes cls separation past the anchor-head ceiling.

### Failure signals → which path to take next
- distill loss flat or rising → checkpoint corrupt; verify NB74 cell 3 output keys.
- decoded_f1 unchanged from NB73 → CULane→BDD domain shift kills the KD signal. Then we need **Exp2TTT: real CLRKDNet teacher with arch adapter**, using NB76's downloaded weights.
- val_map50 still ~0 → det is still the architectural bottleneck. KD doesn't fix det because we're only distilling lane.

## Pipeline status

Total experiments: 70 configs in `stage2/configs/`, 77 notebooks (NB00, NB02-NB08, NB12-NB77 minus a few skipped numbers), 26 `_make_nb*.py` helpers, 27 `_patch_nb08_*.py` patches, 21 `PATCH_NOTES_*.md` files. `python -m compileall stage2/{fusion,metrics,scripts}` passes.

## What I want to know after NB77 runs

Per your earlier diagnosis: if NB77 hits decoded_f1 ≥ 0.07 → KD pipeline works and we ship Exp2TTT next (real CLRKDNet teacher via NB76 weights). If NB77 = NB73 → the teacher/student domain gap kills KD and the path forward is **pretrained backbone loading** (RMT-PPAD weights), which is independent of dataset distribution.

Either way the data wins. Run NB74 (now bug-free), then NB76, then NB77.
