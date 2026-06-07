# P4 migration log

Phase: LaneSegHead wrapper + wire into MTDETRDecoder via `seg_decoder="clr_lane"`.

## 2026-05-25  P4 authored

Per appendix-path3-implementation-prompt.md sec 7.2-7.6, with the deviations
noted below.

### Files touched

| File | Type | Purpose |
|---|---|---|
| `P4_lane_head_integration/tools/lane_seg_head.py` | new | `LaneSegHead` class (standalone, phase-isolated) |
| `P4_lane_head_integration/tools/verify_lane_head_integration.py` | new | acceptance test driver |
| `vendor/RMT-PPAD/ultralytics/nn/modules/lane_head.py` | new | thin shim re-exporting `LaneSegHead` for the ultralytics package |
| `vendor/RMT-PPAD/ultralytics/nn/modules/head.py` | patch | import LaneSegHead; add `seg_decoder=None` param to `MTDETRDecoder.__init__`; branch on it; dispatch the seg call in forward |
| `vendor/RMT-PPAD/ultralytics/nn/tasks.py` | patch | `parse_model` for `MTDETRDecoder` now forwards extra YAML args (e.g. the `"clr_lane"` selector) past the nc_list dict |
| `vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml` | new | lane-only model config: `nc: 2`, `nc_list: {detection: 1, segmentation: 1}`, MTDETRDecoder gets `"clr_lane"` as second arg |

### Deviations from the appendix

1. **`refine_layers=1`, not 3.** Matches CLRKDNet's own CULane config default
   and reduces per-step cost ~3x. Bump in P8 ablation if it helps.
2. **`max_lanes=8`, not 4.** Carried over from P1's decision to keep the
   per-image lane capacity at 8 for BDD scenes.
3. **`seg_decoder` is a real ctor param**, not pulled from a global config.
   `parse_model` forwards it after `(nc, ch, ns_classes)` so older YAMLs
   keep working unchanged.
4. **Forward returns `seg_mask` as a dict** for `clr_lane` mode (just
   `{'lane_output': ...}`). Original return was a tuple
   `(seg_mask_tensor, aux_list)` — keeping the original tuple path
   untouched for the legacy code paths means P5/P7 just type-check the
   seg output.

### Open items

- The model build will require CLRKDNet to be importable (P3 stubs handle
  the CUDA/mmcv shims). On Colab, P3's existing mmcv install + nms_impl
  stub should carry through.
- The eval forward DOES try to construct the full model including
  detection-side modules. If RTDETR's `get_cdn_group` or similar trips on
  the missing `batch` (we pass `batch=None` in eval), we'll need to revisit.

## NB83

`notebooks/stage2_notebook_83_P4_lane_head.ipynb`:
- mounts Drive
- installs `mmcv`
- runs `verify_lane_head_integration.py` via `run_streaming`
- prints the seg_head module subtree for the record

## Pending follow-ups

None for P4 itself. P5 will consume `seg_mask['lane_output']` in
MTDETRDLoss, replacing the drivable + lane mask losses with CLRKDNet's
4-term loss.
