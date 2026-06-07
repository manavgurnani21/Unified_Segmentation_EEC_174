# P8 results

## Ablation table

| Config                        | mAP50 | IoU (lane, rast.) | ACC (lane) | F1 (lane) | Wall (h) |
|-------------------------------|-------|-------------------|------------|-----------|----------|
| baseline_mask (P0)            | 0.849 | 0.568             | 0.847      | n/a       | -        |
| clr_lane_default              | TBD   | TBD               | TBD        | TBD       | TBD      |
| clr_lane_no_square_priors     | TBD   | TBD               | TBD        | TBD       | TBD      |
| clr_lane_no_gca               | TBD   | TBD               | TBD        | TBD       | TBD      |
| clr_lane_attn_h2 (optional)   | TBD   | TBD               | TBD        | TBD       | TBD      |

Fill in by parsing `runs/train/<name>/log.txt` or
`runs/train/<name>/results.csv` after each config finishes.

## Notes from running each config

### clr_lane_default

(fill in after training: smoke pipeline checks, NaN events, gate_mean
range, convergence epoch, anything surprising)

### clr_lane_no_square_priors

(fill in)

### clr_lane_no_gca

(fill in)

### clr_lane_attn_h2

(fill in)

## Failure case analysis

5 sample images where the new lane head fails but the baseline succeeds,
and vice versa. Pairs of inference visualizations + brief diagnosis.

(fill in)

## Inference speed

| Config           | imgsz | batch | FPS (single GPU) |
|------------------|-------|-------|------------------|
| baseline_mask    | 640   | 1     | TBD              |
| clr_lane_default | 640   | 1     | TBD              |

Measured by averaging over 1000 val images on a warm GPU.

## What's lost by removing drivable

The new lane-only model produces no drivable-area output. If a downstream
system needs drivable segmentation, options are:
- Keep the original RMT-PPAD model in parallel as a secondary inference path
- Add drivable back as a separate decoder head (would re-introduce
  `da_seg` into the loss and roughly recover the original 4-task model)

The migration deliberately drops drivable because the hypothesis under
test is "does the curve-regression lane head beat the per-pixel mask
lane head?" — drivable is independent of that question.
