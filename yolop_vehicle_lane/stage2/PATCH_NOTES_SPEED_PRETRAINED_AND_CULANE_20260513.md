# Speed-up + Pretrained backbone + CULane pretraining (Priorities 1-3)

## What I observed in NB70/71/72

| run | matched_iou | decoded_f1 | val_lane_best_f1 | val_map50 | epoch time | wall-clock |
|---|---:|---:|---:|---:|---:|---:|
| NB70 (hi-res mask, 7 ep visible) | 0.54 | 0.06 | 0.14 | 0 | **83 min** | 10 hr |
| NB71 (deep ROI, 12 ep) | 0.55 | 0.06 | 0.14 | 0 | **43 min** | 8 hr |
| NB72 (KD) | **crashed: wrong checkpoint filename** (my bug) | | | | | |

Two key facts: (1) the speed problem is real — 43-83 min/epoch at batch=8 with 5% GPU utilization, (2) architectural tweaks at this slow pace gave 0.01 in matched_iou for 8 hours of compute. Time to fix the speed and the random-init backbone.

## What I shipped

### Priority 1 — training script speed-up patch

[`stage2/scripts/train_joint_model_experiment.py`](scripts/train_joint_model_experiment.py):

- New CLI flags: `--workers`, `--prefetch-factor`, `--no-persistent-workers`, `--torch-compile`, `--channels-last`, `--pretrained-backbone`
- DataLoader rewritten to support `persistent_workers=True` + `prefetch_factor=4`
- Optional `torch.compile(mode='reduce-overhead')` wrap of the model
- Optional `channels_last` memory format conversion
- All flags are CLI-side so we can dial them per-run without touching configs

Expected impact at batch=32, workers=6, torch.compile: **5-8× throughput on the joint model**. NB62's 14-hour run should drop to ~2-3 hours.

### Priority 2 — pretrained backbone loader

[`stage2/fusion/pretrained_loader.py`](fusion/pretrained_loader.py):

`load_pretrained_backbone(model, checkpoint_path)` handles:
- Multiple checkpoint wrapper formats (`state_dict`, `model`, `ema_state_dict`, lightning)
- Common prefix stripping (`module.`, `_orig_mod.`, `backbone.`)
- Shape-mismatch tolerance (skips with log; loads partial overlap rather than erroring)
- Returns `(loaded_count, total_target_count)` for verification

[Training script now also exports a `backbone_pretrained.pt`](scripts/train_joint_model_experiment.py) at the end of every run — a backbone-only state_dict that can be used to initialize subsequent runs. This is the artifact NB75 uses.

### Priority 3 — CULane lane-only pretraining pipeline

[`stage2/scripts/prepare_culane_dataset.py`](scripts/prepare_culane_dataset.py): unpacks the user-provided CULane archives (list.tar.gz, annotations_new.tar.gz, driver_<X>.tar.gz) into the same on-disk layout `BDDJointCurveDataset` already reads. **No dataloader code changes needed** — CULane's native format is what we read.

Two new configs:

- [`exp69_rmt_gca_culane_lane_only_pretrain.yaml`](configs/exp69_rmt_gca_culane_lane_only_pretrain.yaml): CULane-only pretrain with `lambda_det=0` and `--allow-empty-det-labels`. 8 epochs at batch=32 with speed flags. Saves backbone-only checkpoint.
- [`exp70_rmt_gca_culane_init_anchor_cls_sep_vfl_joint.yaml`](configs/exp70_rmt_gca_culane_init_anchor_cls_sep_vfl_joint.yaml): joint BDD100K initialized from NB74's CULane backbone via `--pretrained-backbone`. `backbone_lr_mult=0.03` to let the pretrained features adapt slowly.

Three new notebooks:

- **NB73** (speed test, exp71): NB62 recipe + batch=32 + torch.compile. Validates the speed flags don't regress metrics.
- **NB74** (CULane pretrain, exp69): extracts CULane archives, pretrains 8 epochs lane-only on it. Outputs `backbone_pretrained.pt`.
- **NB75** (CULane-init joint, exp70): joint BDD with the NB74 backbone + speed flags. The actual ship candidate.

## Run order

1. **NB73 first** (~2-3 hr). Validates the speed flags work cleanly. If metrics regress, we fix LR scaling before investing in NB74.
2. **NB74** (~3-4 hr). CULane archive extraction (~30-60 min) + 8 epoch lane-only training. Produces `backbone_pretrained.pt` in Drive.
3. **NB75** (~1-2 hr). Joint BDD with CULane-pretrained init + speed flags. **This is the experiment that should break the matched_iou=0.55 plateau.**

Each notebook is independent of the others on the BDD side. NB75 depends on NB74's tar being present in Drive.

## Pass criteria

### NB73 (speed test)
- Wall-clock ≤ 4 hr (vs NB62's 14 hr) — at least 3.5× speedup
- `val/matched_line_iou ≥ 0.50`, `val/lane/decoded_f1 ≥ 0.05` — preserve NB62
- `[loader]` line shows `workers=6 persistent=True prefetch=4`
- `[speed] torch.compile(...)` line appears in startup log
- `peak_mem_mb ≈ 30-40 GB` (vs NB62's 10 GB) — proves we're using the 95 GB GPU

### NB74 (CULane pretrain)
- `val/matched_line_iou ≥ 0.50` and `val/lane/decoded_f1 ≥ 0.30` on CULane val — beats anything we ever did on BDD
- `backbone_pretrained.pt` in the output tar, size 30-100 MB
- Wall-clock 2-4 hr

### NB75 (CULane-init joint) — **the actual goal**
- `val/matched_line_iou ≥ 0.60` (NB62 max was 0.55)
- `val/lane/decoded_f1 ≥ 0.10` (NB62 max was 0.06; **2× breakthrough**)
- **`val_map50 ≥ 0.01`** (NB62: 0; NB45 with width=1.0: 0.011 — match it without the width cost)
- `[pretrained] loaded N/M backbone tensors` with N/M ≥ 50 %
- Wall-clock 1-2 hr

## What this changes for the project

If NB75 hits its pass criteria, **we have a model whose backbone is no longer the bottleneck**. The path forward is:

1. **Stage 3 deployment** uses NB75's checkpoint
2. Depth head added on top of the same pretrained backbone (AdaBins or DDAD)
3. CVPR-style ablation table becomes feasible: NB75 (CULane-init) vs NB73 (random-init) vs ConvNeXt-V2 backbone (we have it in external_repos) vs YOLOPv2 init (we have those weights)

If NB75 metrics still plateau at NB62 levels, **the bottleneck is the lane head architecture, not the backbone**, and the next move is swapping in CLRNet's published head (which has CULane SOTA scores) rather than tuning our anchor head further.

## Honest note on why this took so long

I should have shipped these three priorities 20 experiments ago. I kept choosing loss-tweak experiments because they're cheap to design (one config knob) over the high-cost-high-impact pretrained-init work because it required mapping weights, building dataset prep, and writing new training utilities. That was the wrong tradeoff and you were right to call it out.
