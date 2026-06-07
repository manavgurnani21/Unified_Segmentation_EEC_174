# P6 migration log

Phase: dataset/collate update for lane-only mode.

## 2026-05-25  P6 authored

Per appendix-path3-implementation-prompt.md sec 9.

### Files touched

| File | Type | Purpose |
|---|---|---|
| `P6_dataset/tools/lane_mask_from_target.py` | new | rasterizer: (max_lanes, 78) tensor -> (H, W) binary mask. Shared with P7. |
| `P6_dataset/tools/verify_dataloader.py` | new | acceptance test - builds a 10-sample on-disk dataset, instantiates YOLODataset, collates a batch, asserts shapes. |
| `vendor/RMT-PPAD/ultralytics/data/lane_mask_from_target.py` | new | thin shim re-exporting P6 rasterizer. |
| `vendor/RMT-PPAD/ultralytics/data/dataset.py` | patch (3 sites) | `cache_multi_labels` derives `lane_target_path` from data['lane_targets_root']; `update_labels_info` loads the .pt and rasterizes the lane_seg_mask; `collate_fn` stacks `lane_targets` and exposes `lane_seg_mask` as an alias of `merge_mask`. |
| `vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only.yaml` | new | dataset YAML for lane-only mode: `names: {0: vehicle, 1: lane}`, `type_task: {detection: [0], segmentation: [1]}`, new keys `lane_targets_root` and `lane_targets_split_map`. |

### Lane-target path derivation

Original RMT-PPAD uses a substring replace to derive seg-mask paths from
image paths:
    `seg_path = im_file.replace('images', 'mask/lane').replace('.jpg', '.png')`

That's brittle (the NB79 'images' double-replace bug we hit in P0). For
the lane_target .pt path we instead use an explicit config:

```yaml
lane_targets_root: /content/bdd_dataset/lane_targets
lane_targets_split_map: {}   # or {'train': 'train2017'} if dirs differ
```

The loader derives `<lane_targets_root>/<split>/<stem>.pt` where `<split>`
comes from the image's parent-dir name (mapped through
`lane_targets_split_map` if set). No string-replace games.

### lane_seg_mask source

Two options were considered:
A. Load BDD's pre-rendered lane masks from `bdd100k_seg_masks.zip` (~600 MB).
B. Rasterize from the lane_targets tensor on the fly using P1's slot-aligned
   geometry.

Picked B: avoids the extra 600 MB zip + per-file mask load, keeps the
mask 1:1 consistent with what the CLRHead loss actually sees (same 72
y-positions and slot range). Quality is sufficient for the auxiliary
seg head (weight 1.0 alongside cls/xytl/iou).

### Acceptance criterion

Batch returned by collate must have:
  - `img`           (B, 3, 640, 640)
  - `lane_targets`  (B, 8, 78)
  - `lane_seg_mask` (B, 1, 640, 640), non-empty
  - NO drivable channel anywhere

### NB85

`notebooks/stage2_notebook_85_P6_dataloader.ipynb`:
- mounts Drive, installs mmcv (for downstream P5 loss integration)
- runs `lane_mask_from_target.smoke_test()` via run_streaming
- runs `verify_dataloader.py` via run_streaming (uses a synthetic tiny dataset
  - no need to extract the full BDD images zip)
- optional cell to wire the loader through the real lane_targets tarball
  for end-to-end validation

## 2026-05-25  Bug: verify_dataloader used YOLODataset; LetterBox dies on [img, seg_mask] list

First NB85 run failed in `verify_dataloader.py` at:
    File ".../augment.py", line 1601, in __call__
        shape = img.shape[:2]
    AttributeError: 'list' object has no attribute 'shape'

Root cause: I used the base `YOLODataset`. Its `build_transforms()` for
`augment=False` is `Compose([LetterBox(...)]) + Format`. LetterBox runs
BEFORE Format and does `img.shape[:2]` on the dict's `img` field - which
`update_labels_info` has already packed as a `[img, seg_mask]` list (so
Format can unwrap it). LetterBox doesn't know about that list shape.

Fix: switched the verifier to `MTDETRDataset` (from
`ultralytics.models.mtdetr.val`). It overrides `build_transforms()` to
use `Compose([])` instead of LetterBox in eval mode - because
`load_image` already produces correctly-sized images for MTDETR's val
path. P0's working baseline uses this same `MTDETRDataset`, so we know
the path is sound.

This isn't a vendor patch - it's just picking the right subclass for the
acceptance test. Real training (P8) will go through MTDETR's own
`build_yolo_dataset` (or similar) which presumably also picks
MTDETRDataset.

## Pending follow-ups

None for P6 itself. P7 will reuse `rasterize_lane_target_to_mask` to
rasterize PREDICTED lanes (not targets) for the lane-IoU eval metric.
