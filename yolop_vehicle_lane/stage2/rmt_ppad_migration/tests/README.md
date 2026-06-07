# Local data-pipeline regression tests

CPU-only tests (numpy / cv2 / torch — **no GPU, no clrkd/ultralytics stack, no
Colab**) of the pure-Python data paths. They guard the bugs that actually bit
this project, so they can be run locally before pushing to Colab.

```
python tests/run_all.py        # exit 0 = all pass; writes tests/RESULTS.txt
```

Each test writes a `*_result.txt` next to itself and returns exit 0 (pass) / 1
(fail). The runner aggregates exit codes (the only output channel that is
reliable in the sandbox; plain stdout gets garbled).

| Test | Guards | The bug it pins |
|---|---|---|
| `test_curve_f1.py` | curve-F1 / curveIoU metric | broad-pool curveIoU (top-64, not top-8) + preds not dropped by `row[1]<0.5` (the curveIoU=0 fixes) |
| `test_lane_rasterize.py` | eval rasterizer + GT rasterizer | PRED `length*n_strips` scaling (the frozen-IoU breakthrough) + PRED/GT encoding agreement |
| `test_drivable_extract.py` | `aux_seg/tools/extract_drivable_subset.py` | id-mask-over-colormap `_rank`, 0/1/2→binary, stem matching, coverage, missing-handling (NB98's 4%/false-positive risks) |
| `test_notebook_decode.py` | NB101 + NB102 inline `decode_lanes`/`read_boxes` | drift of the copied inspection-cell decode vs the canonical rasterizer |
| `test_bezier_trange.py` | `bezier_ops.render_with_validity` | t-range honored (full curve vs 5% stub) — the bezier-head init fix rationale |

## Notes
- `test_drivable_extract.py` builds a synthetic zip mimicking BDD100K "Drivable
  Maps" (`labels/.../<stem>_drivable_id.png` + `color_labels/...`) and runs the
  real extractor end-to-end, then asserts the written mask is the binarized id
  mask (not the colormap) at the correct geometry.
- All tests are deterministic (seeded). Re-running is idempotent (each rebuilds
  its own scratch under `tests/_work_*`).
- Last full run: **5/5 passed** (see `RESULTS.txt`).
