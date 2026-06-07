# P8 ablation matrix

The migration's final acceptance is a 4-row ablation table (appendix-path3
sec 11.4). Each row is a 250-epoch run; full table = ~12-20 GPU-days on
3×24GB. The smoke test (NB87) verifies row 2's config trains 2 epochs
without errors before you commit the long runs.

## Configurations

| # | Name                          | YAML delta                                                                                                                                        | Question being answered |
|---|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| 1 | `baseline_mask`               | original `rtdetr-l_bdd.yaml` (TransformerSegmentationDecoder, drivable + lane mask) — already trained as P0 reference                              | What we are improving on |
| 2 | `clr_lane_default`            | `rtdetr-l_bdd_clr_lane.yaml` as-is: GCA on, `CLRHeadForSquareImage` (32/128/32 priors), `max_lanes=8`                                              | Does the full pipeline help over the mask baseline? |
| 3 | `clr_lane_no_square_priors`   | YAML: `rtdetr-l_bdd_clr_lane_no_square_priors.yaml` (passes `"no_square_priors"` to MTDETRDecoder). LaneSegHead instantiates base `CLRHead` (24/144/24 priors) instead of `CLRHeadForSquareImage` | Does retuning priors for 1:1 matter, or is the default CULane prior layout fine? |
| 4 | `clr_lane_no_gca`             | YAML: `rtdetr-l_bdd_clr_lane_no_gca.yaml` (passes `"no_gca"`). `MTDETRDecoder._get_encoder_input` sends raw `input_proj` features to LaneSegHead instead of `gated_seg` (task_adapter_seg + gate_seg bypassed for the lane path; still computed for the det path) | Does GCA help the lane branch, or are raw FPN features sufficient? |
| 5 | `clr_lane_attn_h2` (deferred) | Not implemented yet - requires editing CLRKDNet's `roi_gather.py` (attention head count). Run only if rows 2-4 leave the headline IoU within 1% and we want one more lever to ablate.                                  | Is the 8-head attention overkill for lane regression? |

## How to launch a config

The Colab orchestrator is `notebooks/stage2_notebook_88_P8_full_ablation.ipynb`.
For standalone use:

```bash
# Row 2 (default)
python stage2/rmt_ppad_migration/P8_train/scripts/train_lane_only.py \
    --mode full \
    --name clr_lane_default \
    --project /content/runs \
    --model-yaml stage2/rmt_ppad_migration/vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml \
    --device 0 --batch 8

# Row 3 (no_square_priors): same as row 2 but with the no_square_priors YAML
python ... --name clr_lane_no_square_priors \
    --model-yaml .../rtdetr-l_bdd_clr_lane_no_square_priors.yaml

# Row 4 (no_gca)
python ... --name clr_lane_no_gca \
    --model-yaml .../rtdetr-l_bdd_clr_lane_no_gca.yaml
```

All three runs use the same training script and same data YAML
(`BDD_lane_only.yaml`). Only the model YAML changes. The model YAML's
3rd positional arg to `MTDETRDecoder` (`"no_square_priors"` or
`"no_gca"`) selects which ablation branch fires; `parse_model` forwards
the arg verbatim.

Row 5 (`attn_h2`) requires editing `ROIGather`'s attention layer
construction inside CLRKDNet's `clrkd/models/utils/roi_gather.py`.
Deferred until rows 2-4 are in.

## Results template

After each row completes, parse `runs/train/<name>/log.txt` (or the JSON
written by `train_lane_only.py`) and fill in `P8_train/results.md`:

| Config                       | mAP50 | IoU (lane, rast.) | ACC (lane) | F1 (lane) | Wall (h) |
|------------------------------|-------|-------------------|------------|-----------|----------|
| baseline_mask (P0)           | 0.849 | 0.568             | 0.847      | n/a       | -        |
| clr_lane_default             |       |                   |            |           |          |
| clr_lane_no_square_priors    |       |                   |            |           |          |
| clr_lane_no_gca              |       |                   |            |           |          |
| clr_lane_attn_h2 (optional)  |       |                   |            |           |          |
