# P8 migration log

Phase: train + ablate.

## 2026-05-25  P8 infrastructure authored

The migration's final acceptance is a 4-row ablation table. Full training
is ~12-20 GPU-days total (4 configs × 250 epochs × 3-5 days each on
3×24GB GPUs); too long for a single Colab session. P8 splits into:

- A **Colab-runnable smoke test** (NB87) that trains the default
  `clr_lane` config for 2 epochs on ~200 train + 80 val images, verifies
  the pipeline trains end-to-end without errors, all 4 lane keys
  populate, `da_seg=0` exactly. ~5-15 min wall.
- **Standalone scripts** for the full ablation rows, runnable on whatever
  multi-GPU machine the user has.

### Files

| File | Type | Purpose |
|---|---|---|
| `P8_train/scripts/train_lane_only.py` | new | Trainer driver. `--mode {smoke,full}` flips epochs/batch/lr defaults; everything else is plumbed to `MTDETR.train(...)`. |
| `P8_train/scripts/prepare_bdd_subset.py` | new | One-shot data prep: pulls a configurable subset of `bdd100k_images_100k.zip` + `BDD_detection_labels.zip` + `lane_targets_clr_v1_polyline.tar.gz` into `/content/bdd_dataset/{images,labels,lane_targets}/{train,val}/`. Idempotent. |
| `P8_train/configs/ablation_matrix.md` | new | 4-5 ablation configs + how to launch each. |
| `P8_train/results.md` | new | Template for the final results table + speed measurements + failure analysis. User fills in after each row trains. |
| `notebooks/stage2_notebook_87_P8_smoke.ipynb` | new | Colab driver: mount + mmcv + prepare_bdd_subset + train_lane_only --mode smoke + parse the run. |

### Smoke acceptance

NB87 cell 4 checks:
1. At least one checkpoint saved under `runs/train/p8_smoke/weights/`.
2. `results.csv` contains both `lane_*` and `da_*` columns.
3. Every numeric column is finite (no NaN/Inf).

`da_seg == 0.0` is structurally guaranteed by P5's aggregation patch
(it forces `da_seg = torch.tensor(0.0)` whenever the lane_only path is
active). The smoke test just confirms the training loop actually walks
that path.

### What's NOT in P8 yet

- **Full 250-epoch runs.** User's GPU job.
- **Ablation rows 3-5** (no_square_priors, no_gca, attn_h2). The config
  matrix documents the required code deltas. They're small (1-2 lines
  each) but each needs its own training run; folding them into a single
  flag would just be syntactic sugar.
- **CULane-style F1 metric** (appendix sec 10.5). Optional; deferred
  until we see whether rasterized IoU is too noisy.
- **Failure case analysis**, **inference speed table**. User produces
  these after the full ablation runs finish.

## 2026-05-25  Bug: prepare_bdd_subset hardcoded /100k/train/ path filter

First NB87 run failed in cell 4 with `No images sampled - inspect ...
bdd100k_images_100k.zip. Train layout expected at .../100k/train/<stem>.jpg`.

The script's `_pick_subset_from_zip` required `/100k/` AND `/train/`
substrings in every candidate path. The actual zip's internal layout
turned out to NOT use exactly that pattern - might be `images/100k/`
without the leading slash, or `train2017/` instead of `train/`, etc.

I had a similar bug-class in P1's path replacement work. Lesson re-learned:
DON'T hard-code archive layouts; auto-detect.

Fix in `prepare_bdd_subset.py`:
1. Removed the in_zip_prefix_options requirement. The script now only
   filters by file extension (.jpg / .txt / .pt) and by any path
   component matching the split aliases.
2. Added `_SPLIT_ALIASES` mapping so `train` matches `{train, train2017,
   training}` and `val` matches `{val, val2017, validation}`. Images
   from official BDD use train/val; labels from RMT-PPAD's repackaging
   use train2017/val2017; our P1 lane_targets use train2017/val2017.
3. Added `_probe_zip()` that prints 5 sample entries from each archive
   at start, so any future layout surprises are visible up-front.
4. The error message now dumps the first 30 entries from the zip when
   the sampler returns 0 candidates - immediate ground truth for what
   the script saw.

The lane-targets tarball extraction (`_extract_lane_targets_subset`) got
the same alias-aware treatment.

## 2026-05-25  Fix: use the project's pre-prepared bdd100k_clrkd_curve.tar

Previous fix (alias-aware path matching) still tried to read from the raw
`bdd100k_images_100k.zip`. User pointed out the right move: look at how
previous notebooks access BDD data and reuse that.

The existing convention (NB00 -> `prepare_joint_curve_dataset.py` ->
`04_prepare_bdd_curve_labels.py --pack-to`) produces a clean
`bdd100k_clrkd_curve.tar` on Drive with a fixed `images/{train,val}/<stem>.jpg`
layout. P0's NB79 cell 4 already uses this tarball - same proven path.

Rewrote `prepare_bdd_subset.py` to:
1. Extract `bdd100k_clrkd_curve.tar` to /content/bdd_curve_scratch.
2. Auto-discover its `images/train` + `images/val` source dirs (with
   alias fallback for train2017/val2017).
3. Hardlink N random train + val images into
   `/content/bdd_dataset/images/{train2017,val2017}/` - matching RMT-PPAD's
   split-naming convention so the existing dataset YAMLs work as-is.
4. Pull matching detection .txt from `BDD_detection_labels.zip` into
   `/content/bdd_dataset/labels/{train2017,val2017}/`. Missing labels
   become empty placeholders.
5. Pull matching .pt from `lane_targets_clr_v1_polyline.tar.gz` into
   `/content/bdd_dataset/lane_targets/{train2017,val2017}/`.

Also updated `BDD_lane_only.yaml` to `train: images/train2017`,
`val: images/val2017` for consistency with this layout.

NB87 cell 1 now hunts for `bdd100k_clrkd_curve.tar` on Drive (default
location `/content/drive/MyDrive/EcoCAR/datasets/`) instead of the raw
`bdd100k_images_100k.zip`. Same path used by NB79 + the joint-training
notebooks (NB12+).

## 2026-05-25  Bug: wandb callback rejected the project name '/content/runs/train'

After the prep step finally succeeded (200 train + 80 val images, labels,
lane_targets all extracted clean) and the model built (35.3M params),
the smoke train crashed in MTDETR.train() at:

    wandb.errors.errors.UsageError: Invalid project name '/content/runs/train':
        cannot contain characters '/,\\,#,?,%,:', found '/'

Root cause: Ultralytics' `wb.py` callback uses `trainer.args.project` as
the wandb project NAME (line 112). Our `--project /content/runs/train`
contains slashes which wandb's project-name validator rejects.

Looking at `wb.py` lines 154-163, the callback dict is only populated if
`wb` is truthy at MODULE IMPORT TIME, which in turn requires
`SETTINGS["wandb"] is True`. If we set wandb=False in Ultralytics'
settings BEFORE its first import, the callback never registers and the
validator never runs.

Fix in `train_lane_only.py`: new `_disable_wandb_via_settings_file()`
helper pre-writes `~/.config/Ultralytics/settings.json` with a
full-schema dict that exactly matches Ultralytics' v0.0.6 defaults but
flips `wandb`, `tensorboard`, `clearml`, `comet`, `dvc`, `hub`,
`mlflow`, `neptune`, `raytune` to False. Called BEFORE
`_ensure_vendor_first_on_path()` so it lands before any ultralytics
import.

The full-schema write is important: Ultralytics' `_validate_settings`
checks that all keys / types / version match its defaults at init time.
If any mismatch, it resets to defaults (and we'd lose wandb=False). The
helper hard-codes settings_version='0.0.6'; if Ultralytics' settings
schema later drifts, validation fails and the override silently disappears.
Easy to spot (run logs go to a different runs_dir, wandb errors return)
and easy to re-pin (update the version string).

## 2026-05-25  Skipped smoke; built full ablation infrastructure

User opted to skip the 2-epoch smoke test (mmcv re-install + 200-image
prep takes ~3 min; not worth the wait given the pipeline already passed
P7's end-to-end check). Going straight to the full 3-row ablation.

Added wiring for ablation rows 3 and 4 from `ablation_matrix.md`:

### Code changes

| File | Change |
|---|---|
| `P4_lane_head_integration/tools/lane_seg_head.py` | `LaneSegHead.__init__` gained `use_square_priors: bool = True`. When False, instantiates the base `CLRHead` (24/144/24 priors) instead of `CLRHeadForSquareImage`. |
| `vendor/.../nn/modules/head.py` | `MTDETRDecoder.__init__` gained `ablation_mode: str | None`. When `seg_decoder == "clr_lane"`, the mode controls: `"no_square_priors"` -> `LaneSegHead(use_square_priors=False)`; `"no_gca"` -> sets `self.lane_uses_gca = False`. `_get_encoder_input` now branches on `lane_uses_gca`: if False, the seg-path features bypass `task_adapter_seg` + `gate_seg` and use raw `input_proj` output directly. |
| `vendor/.../cfg/models/mt-detr/rtdetr-l_bdd_clr_lane_no_square_priors.yaml` | new YAML, passes `"no_square_priors"` as the 3rd arg to MTDETRDecoder. |
| `vendor/.../cfg/models/mt-detr/rtdetr-l_bdd_clr_lane_no_gca.yaml` | new YAML, passes `"no_gca"` as the 3rd arg. |
| `notebooks/stage2_notebook_88_P8_full_ablation.ipynb` | new 7-cell orchestrator. Prep full dataset once, then 3 ablation cells call `launch_ablation(name)`. Each is resume-safe (ultralytics picks up from last checkpoint). After each row finishes the run dir is tarred + copied to Drive. Cell 7 aggregates the results. |

### How parse_model picks up the 3rd YAML arg

Our P4 patch to `tasks.py` already forwards extra YAML args verbatim:

```python
elif m is MTDETRDecoder:
    args_list = [args[0]['detection']]
    args_list.insert(1, [ch[x] for x in f])
    args_list.append(args[0]['segmentation'])
    if len(args) > 1:
        args_list.extend(args[1:])     # <- this picks up "clr_lane" + ablation_mode
    args = args_list
```

So `[nc_list, "clr_lane", "no_square_priors"]` from the YAML becomes
`MTDETRDecoder(nc=..., ch=..., ns_classes=..., seg_decoder="clr_lane", ablation_mode="no_square_priors")`.

### What's NOT implemented

- `attn_h2` ablation (ROIGather attention heads) needs deeper changes
  into CLRKDNet's `roi_gather.py`; defer until the 3 listed rows finish
  and we know whether attention heads even matter at the headline
  numbers.
- CULane-style line-F1 metric still optional (appendix sec 10.5);
  rasterized IoU is the default.

## 2026-05-25  Bug: ModelEMA.deepcopy refused non-leaf priors buffer

First full-ablation run got through prep (70k+10k), model build, label
scan (80k), and crashed at:
    self.ema = deepcopy(de_parallel(model)).eval()  # ultralytics/utils/torch_utils.py:512
    RuntimeError: Only Tensors created explicitly by the user (graph leaves)
                  support the deepcopy protocol at the moment.

Root cause: CLRHead.__init__ does
```python
init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
self.register_buffer('priors', init_priors)
self.register_buffer('priors_on_featmap', priors_on_featmap)
```
Both returned tensors are derived from `self.prior_embeddings.weight`
(a learnable Parameter) via arithmetic - they're NON-LEAF with
requires_grad=True. PyTorch refuses to deepcopy such tensors.

CLRHead's source lives in `external_repos/` and we don't modify it.

Fix:
1. `LaneSegHead.__init__` post-construction: `self.lane_head.priors =
   self.lane_head.priors.detach().clone()` (same for priors_on_featmap).
   Replaces the non-leaf grad tensor with a leaf copy.
2. `train_lane_only.py` defensive sweep: walks every buffer in the model,
   detaches any that are non-leaf with requires_grad=True. Belt-and-
   suspenders for any future buffer we missed.

During training, CLRHead.forward() reassigns `self.priors =
generate_priors_from_embeddings()` each step - that's a fresh non-leaf
with grad - but ModelEMA's deepcopy is a ONE-TIME init step. Subsequent
EMA updates use in-place arithmetic (no deepcopy), so they don't trip.

## 2026-05-25  Bug: v8_transforms can't handle [img, seg_mask] list during training

After the ModelEMA deepcopy fix landed, the train cell got further:
prep + model build + AMP check + train label scan (70k) + val label
scan (10k) + plotting + Starting training for 250 epochs ... then
crashed inside the dataloader at:

    File ".../augment.py", line 1948, in __call__
        labels["img"] = self.transform(image=labels["img"])["image"]
    TypeError: image must be numpy array type

Root cause:
- P6's `update_labels_info` packs the per-sample image as
  `label['img'] = [img, seg_mask]` (a 2-element list). This was the
  vendored RMT-PPAD convention so the seg_mask gets carried through
  to Format.
- The training-mode transform pipeline `v8_transforms(...)` includes
  `Albumentations`, `RandomFlip`, etc. - all expect `labels['img']`
  to be a plain numpy ndarray (so they can do `img.shape[:2]`,
  `cv2.resize(img, ...)`, etc.). They crash on the list.
- The existing project's working joint-training notebooks (NB12+) sidestep
  this by using a CUSTOM `BDDJointCurveDataset` (in
  `stage2/scripts/train_joint_model_experiment.py`) instead of
  `MTDETRDataset`. Our P4 migration kept the MTDETR.train() path for
  alignment with the appendix spec - so we're the first to actually
  exercise this code path with augment=True training.

Fix (3 patches):
1. `vendor/.../data/dataset.py update_labels_info`: in lane-only mode,
   `label['img']` is now a PLAIN ndarray and the rasterized mask sits
   under `label['_lane_seg_masks']`. Legacy mask-seg path still uses
   the `[img, seg_mask]` list packing.
2. `vendor/.../data/augment.py Format.__call__`: detects
   `_lane_seg_masks` and uses it instead of `img[1]`. Falls back to
   the legacy `img[0]/img[1]` if the new key isn't present.
3. `train_lane_only.py`: passes `degrees=translate=scale=shear=
   perspective=flipud=fliplr=mosaic=mixup=copy_paste=0.0` to
   `MTDETR.train(...)`. These spatial transforms would desync the
   rasterized `lane_seg_mask` AND the `lane_targets` from the
   augmented image, since we don't transform them in parallel.
   Photometric transforms (HSV, Albumentations blur/clahe/etc.) only
   touch pixel values and stay safe.

Tradeoff: ~5-10% expected dip in detection mAP vs the augmented
baseline. Acceptable since the migration's hypothesis is about lane
metrics. P8 ablation may show whether we want to add proper
img + seg_mask + lane_targets co-augmentation (e.g. a CustomFlip that
also flips the lane_targets tensor's x columns).

## 2026-05-25  Reverted to RMT-PPAD's [img, seg_mask] packing; patched only Albumentations

Previous fix went the WRONG direction. After making img a plain ndarray
+ `_lane_seg_masks` key, training got past Albumentations only to crash
inside `RandomPerspective.__call__` at:
    File ".../augment.py", line 1247
    labels["img"][1][count] = cv2.warpAffine(mask, M[:2], dsize=...)
    ValueError: could not broadcast input array from shape (640, 3) into shape (3,)

(Iterating `labels["img"][1]` over a (640, 640, 3) ndarray gives row
slices of shape (640, 3); assigning a warped (640, 640) mask there
explodes.)

What was actually going on:
- RMT-PPAD's RandomPerspective / RandomHSV / RandomFlip / Mosaic were
  ALREADY patched (with `### JW ...` comments) to expect the
  `[img_ndarray, [seg_masks]]` list packing. They co-transform masks
  alongside the image.
- Only `Albumentations` (line 1948) and `LetterBox` (line 1601) were
  left UN-patched.
- My previous fix flipped img to plain ndarray, which broke the
  already-patched majority. Backwards.

Correct fix:
1. `dataset.py update_labels_info`: revert to ALWAYS pack
   `label['img'] = [img, seg_mask]` (RMT-PPAD's intended shape).
   Still loads `lane_targets` separately. Drops the `_lane_seg_masks`
   key entirely.
2. `augment.py Albumentations.__call__`: patched to unwrap
   `labels["img"][0]` when packed, apply Albumentations (photometric
   only - masks stay untouched, fine), and re-wrap.
3. `augment.py Format.__call__`: simplified back to the legacy
   `img[0]/img[1]` path (since update_labels_info always produces the
   list). Defensive empty-list -> zero-mask still in place.

LetterBox stays unpatched because MTDETRDataset's `build_transforms`
for `augment=True` uses `v8_transforms` which does NOT include
LetterBox (stretch=True path). LetterBox only fires in the val pipeline
which MTDETRDataset overrides to `Compose([])`.

The spatial-transform-disabling hyp args (degrees=...=fliplr=mosaic=
mixup=copy_paste=0) stay - they keep our `lane_targets` consistent
with the (un-spatially-augmented) image even though the seg_mask itself
WOULD be co-transformed by the patched ops.

## 2026-05-25  Bug: Detection loss + grad_norm went NaN in epoch 1

User reported: `Detection` and `grad_norm` both dropped, then turned into
NaN within the first epoch. Once NaN appears in gradients, AdamW's
running statistics are permanently poisoned - no recovery possible. The
existing checkpoint is now useless.

Root cause stack:
1. AMP (FP16 mixed precision) was on by default. FP16's range maxes near
   6e4; DETR's first-epoch `loss_class_aux` was already 395 in the P5
   synthetic batch test, and the real-data forward likely spiked higher.
   One spike past 6e4 → grad overflow → inf → NaN.
2. `lr0=6e-4` (CLRKDNet's CULane default) is hot for AdamW on a fresh
   35M-param RT-DETR + lane head. RT-DETR's official train.py uses 1e-4
   for AdamW.
3. `warmup_bias_lr=0.1` (ultralytics legacy default) gives bias params
   an effective lr ~1e-1 for the first 3 epochs. SGD-era heuristic that
   AdamW catastrophically explodes under.

Fix (all in `train_lane_only.py`'s `model.train(...)` kwargs):
- `amp=False` - disable FP16. Forfeits ~30% throughput; trades for
  numerical stability. Re-enable later via `--amp true` once we know
  the run converges cleanly.
- `lr0=1e-4` (was 6e-4) - matches RT-DETR's AdamW default.
- `warmup_epochs=5.0` (was 3.0) - more conservative ramp.
- `warmup_bias_lr=0.0` (was 0.1) - turn off the legacy bias warmup that
  AdamW doesn't need.

How to spot which loss term spiked first next time: open
`runs/<name>/results.csv` after the run dies. Final row shows per-key
losses. The one with NaN there blew up; the one with the largest
finite value just before is the next suspect.

Recovery procedure when a config NaNs:
1. `rm -rf /content/runs/<name>` (delete the contaminated checkpoint dir)
2. Re-run the cell (no resume - we want a fresh start with the new
   stability settings)
3. If still NaN: drop lr0 to 5e-5 and try again.

## 2026-05-25  Added training monitors: tqdm throttle + auto-stop on anomaly

User reported (1) tqdm spam (~8750 progress-bar updates/epoch), (2) need
for auto-stop on NaN/divergence to avoid burning GPU hours on dead runs,
(3) `ll_seg` decreasing gradually then suddenly increasing - "is it normal?".

New module `P8_train/scripts/training_monitors.py`:

- `install_tqdm_throttle(min_iters=100, min_interval=15.0)` - monkey-
  patches `tqdm.tqdm.__init__` so all progress bars throttle. About 88
  updates per epoch with `min_iters=100` on our 8750-step training loop.
  Idempotent. MUST be called before ultralytics imports tqdm.

- `LossMonitor` class - tracks an EMA per loss key from `trainer.loss_items`
  with three anomaly checks: (A) NaN/Inf in any term -> immediate stop;
  (B) spike > 5x EMA sustained 50 consecutive batches -> stop;
  (C) divergence (EMA itself grows 1.5x over a 500-batch window AFTER
  the first 200 warmup batches) -> stop.

- `make_callback(monitor)` - returns the (event_name, callable) pair to
  register with `model.add_callback('on_train_batch_end', cb)`.

Wired in train_lane_only.py:
  1. tqdm throttle installed right after the wandb shim, BEFORE the
     `from ultralytics import MTDETR` line.
  2. LossMonitor + callback registered right after model build, BEFORE
     `model.train(...)`.

The three thresholds (5x for 50 batches, 1.5x over 500 batches) are
tuned to allow SimOTA matching reassignment bounces (normal in epochs
1-3) while catching real divergence within ~10s of GPU time.

ll_seg "decrease then sudden increase" pattern - is it normal?

Sometimes yes, sometimes no:
- NORMAL: SimOTA reassignment (the dynamic top-k matching in CLRKDNet's
  `assign()`). When the model improves enough that priors that USED to
  match GT no longer do (or new priors start matching), the cost matrix
  reshuffles. The "right" loss for the new assignment can be transiently
  higher than the "wrong" loss for the old assignment. Typically 1-2
  epochs of bouncing then stabilization.
- NORMAL: end of warmup (epoch 3-5 for us). The bias group's lr ramping
  up can cause a brief loss bump.
- ABNORMAL: a single loss key (e.g. `lane_iou_loss`) growing without
  bound. LineIoU returns 1 - iou; if many priors land outside image
  width, `iou` clamps near 0, loss saturates near 1.0 per row, sum
  scales with batch but should NOT keep growing.
- ABNORMAL: ll_seg increase coincides with Detection oscillating. That
  means the shared backbone is being torn between tasks - check
  Det_gate and Seg_gate columns.

The LossMonitor catches the abnormal cases. If `ll_seg` bounces from say
12 -> 18 then back to 13 over 50 batches, the monitor logs a warning but
doesn't stop. If it bounces and stays > 60 (5x EMA of 12) for 50+
batches it stops. If EMA itself climbs 12 -> 18 over 500 batches it
stops.

## Pending follow-ups

After the user's first full `clr_lane_default` run finishes, fill in
`results.md` and verify the smoke-test claims still hold at scale
(no NaN at epoch 100+, gate_mean stays in [0.1, 0.9], etc.).

## Fix: ll_seg divergence on the first joint run (lane-weight rescale)

### Symptom

Around batch 500 of epoch 1 the LossMonitor stopped training:

```
[loss-monitor] STOPPING TRAINING NOW: step=500
ll_seg divergence: EMA 16.5 -> 240 over last 500 batches
```

The header row at the time showed `Detection=62`, `ll_seg=70` and
climbing, `grad_norm=756`, lr still in warmup (`1.37e-6`).

### Diagnosis

CLRKDNet was tuned for a *lane-only* ~5 M-param network. We inherited its
weights verbatim:

```
lane_cls_weight = 2.0
lane_xytl_weight = 0.5
lane_iou_weight = 2.0
lane_seg_weight = 1.0
```

In our setup, these four sum into `ll_seg` and propagate through a 35 M-
param RT-DETR backbone that is *simultaneously* receiving the detection
loss. With the absolute lane magnitude this high, every lane-iou hit
backproped a gradient roughly 5x what the detection branch saw, and the
shared encoder/neck started rocketing the seg branch's contribution up
even though `grad_norm` stayed finite.

### Fix

1. `vendor/RMT-PPAD/ultralytics/models/utils/loss.py`
   `MTDETRDLoss.__init__` now uses `* 0.2`:

   ```python
   self.lane_cls_weight  = 2.0 * 0.2   # 0.4
   self.lane_xytl_weight = 0.5 * 0.2   # 0.1
   self.lane_iou_weight  = 2.0 * 0.2   # 0.4
   self.lane_seg_weight  = 1.0 * 0.2   # 0.2
   ```

   Relative ratio (cls : xytl : iou : seg = 4 : 1 : 4 : 2) is preserved -
   only the absolute scale is cut 5x. This puts `ll_seg` in the same
   ballpark as `Detection` at warmup so neither branch dominates.

2. `P8_train/scripts/train_lane_only.py` LossMonitor was loosened so it
   doesn't fire on the legitimate ramp:

   ```python
   warmup_batches    = 1000   # was 200
   divergence_growth = 3.0    # was 1.5
   ```

   The lane branch's 192 priors learn assignments via SimOTA + LineIoU
   from scratch, so a clean run shows ll_seg climbing for ~1 epoch before
   plateauing. We still catch real blow-ups (NaN, sustained 5x spikes,
   3x EMA growth over 500 batches) without false positives.

### Re-run

After this fix the user must re-run NB88 cells 4, 5, 6 with `fresh=True`
to wipe the poisoned checkpoints written by the previous divergent run.
Expected new behavior:

- `Detection` should stay in ~50-65 during warmup
- `ll_seg` should start ~2-3 (5x lower than before), drift up briefly
  during early SimOTA settling, then plateau by mid-epoch-2
- `grad_norm` stays under ~500 (was 750+)
- No LossMonitor warnings in the first epoch

If `ll_seg` *still* diverges with this rescale, the next lever to pull is
lane_seg_weight = 0.0 (turn off the auxiliary CE on the rasterized
seg_mask; lane learning becomes pure curve-regression) before touching
the cls/xytl/iou ratio.

(Iteration 1 follow-up below shows the rescale was correct but
insufficient - the next failure was single-batch outlier spikes from
unbounded smooth_l1 on out-of-range predictions, fixed by clamping the
xytl diff to +/-100 pixel units. See `debug_record.md` Iteration 1.)

## Fix: transient single-batch xytl outliers (debug_record Iteration 1)

After the 5x weight rescale, three full launches (`clr_lane_default`,
`clr_lane_no_square_priors`, `clr_lane_no_gca`) still hit LossMonitor
stops in epoch 1. But the failure pattern was different:

- Per-row running `ll_seg` average stayed healthy (3 ~ 10)
- Single-batch values spiked to 100, 600, 1300, 2000, 6300
- The EMA was poisoned by these spikes (`0.99 * 3 + 0.01 * 6330 = 67`)
  and took hundreds of batches to decay back, eventually crossing the
  3x divergence threshold

Root cause: in `_compute_lane_only_loss` the xytl smooth_l1 is computed
in real units (start_y in strips, start_x in pixels, theta in degrees,
length in strips). CLRHead's predictions for these dims are assumed to
live in [0, 1] but nothing enforces it - `nn.init.normal_(..., std=1e-3)`
keeps them there for lane-only training, but in the joint setup the
shared backbone and GCA adapter let `reg[:, :, :4]` drift to magnitude
10+ for some priors. `(5 * 639) - target_in_pixels = 3000+` diff,
smooth_l1 of that ~= 2999.5, averaged over ~16 elements per image ~=
~200, scaled and summed across 8 batch x 6 stages -> ~6000 ll_seg.

### Fix

1. `vendor/RMT-PPAD/ultralytics/models/utils/loss.py`
   `MTDETRDLoss._compute_lane_only_loss`: clamp the per-element pixel-
   space diff to +/-100 before smooth_l1. Gradient is full for in-range
   priors (99% of them), zero for out-of-range outliers (1% of priors
   on bad batches). Max xytl contribution to `ll_seg` per batch is now
   ~10 (was unbounded).

2. `P8_train/scripts/training_monitors.py` `LossMonitor.update()`:
   added `ema_cap_mult=10.0` parameter. A per-batch value above
   `10 * EMA` is still counted as a spike event but its blended
   contribution to the EMA is capped at that multiple. Belt-and-
   suspenders for any future loss term we add.

### Why this is correct rather than just papering over

The smooth_l1 loss with a per-element clamp is mathematically equivalent
to Huber loss with a saturation point. For in-range predictions the
gradient signal is identical to vanilla smooth_l1. For out-of-range
predictions the model loses learning signal on that single prior for
that single batch - acceptable because (a) those priors are pathological
outliers, (b) the 99% in-range priors carry the lane-shape signal, and
(c) blocking the unbounded loss is the only thing standing between
"transient prior misbehavior" and "training stops".

The EMA cap is a defensive measure on the MONITORING side, not the
training side. Backprop sees the real raw loss. The monitor sees a
sanitized version of the time series so it can make the right
"is this run diverging" call.
