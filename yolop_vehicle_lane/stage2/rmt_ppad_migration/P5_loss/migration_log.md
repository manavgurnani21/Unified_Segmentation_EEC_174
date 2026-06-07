# P5 migration log

Phase: port CLRKDNet loss utilities + add `lane_only_mode` to `MTDETRDLoss`.

## 2026-05-25  P5 authored

Per appendix-path3-implementation-prompt.md sec 8.

### Files touched

| File | Type | Purpose |
|---|---|---|
| `P5_loss/tools/lane_losses.py` | new | port: `line_iou`, `liou_loss`, `distance_cost`, `focal_cost`, `dynamic_k_assign`, `assign`, `FocalLossForLane` |
| `P5_loss/tools/verify_lane_loss.py` | new | acceptance test: build clr_lane model, run loss on synthetic batch, assert da_seg=0 + finite + backward OK |
| `vendor/RMT-PPAD/ultralytics/models/utils/lane_losses.py` | new | thin shim re-exporting the P5 module |
| `vendor/RMT-PPAD/ultralytics/models/utils/loss.py` | patch | `MTDETRDLoss.__init__` gains `lane_only_mode`; forward branches; new `_compute_lane_only_loss` method emits 4 lane keys |
| `vendor/RMT-PPAD/ultralytics/nn/tasks.py` | patch | `init_criterion` detects `seg_decoder_mode=='clr_lane'` and passes `lane_only_mode=True`; `loss()` reads `lane_seg_mask`+`lane_targets` from batch in that mode; aggregation collapses 4 lane keys into `ll_seg` and forces `da_seg=0.0` |

### Loss term schema (lane-only mode)

  `lane_cls_loss`      = FocalLossForLane on (B*num_priors, 2) prior classification, weight 2.0
  `lane_xytl_loss`     = smooth-L1 on (start_y, start_x, theta, length), weight 0.5
  `lane_iou_loss`      = (1 - line_iou) on the 72 normalized x-offsets, weight 2.0
  `lane_seg_aux_loss`  = cross-entropy on CLRHead's internal 2-channel aux seg map vs `lane_seg_mask`, weight 1.0

All weights from CLRKDNet's `configs/DLA_CULane.py`.

### Acceptance criterion

`da_seg == 0.0` exactly + all four lane keys present + finite + backward OK.

### Deviations from the appendix

- The appendix's `_compute_lane_only_loss` reads `seg_output['lane_output']`
  directly; in our impl `seg_mask` arrives as a 2-element list
  `[seg_masks, aux_list]` from `tasks.py:loss()` (kept that interface
  intact to minimize blast radius). We index `seg_mask[0]` to get the
  LaneSegHead dict.
- Eval-mode safety: if `lane_output` is a raw tensor (eval, not training),
  we skip loss computation and return all zeros so val.py doesn't crash
  on shape mismatch. The real eval-mode metrics come from P7's validator.
- `n_strips` is read from the prediction tensor shape so the formulas
  still work if num_points ever changes from 72.

### NB84

`notebooks/stage2_notebook_84_P5_loss.ipynb`:
- mounts Drive, installs mmcv
- runs `verify_lane_loss.py` via `run_streaming`
- in-process re-run of the loss to dump per-key values

## Pending follow-ups

None for P5 itself. P6 will deliver the real dataloader that produces
`lane_seg_mask` and `lane_targets` from the .pt files P1 wrote.
