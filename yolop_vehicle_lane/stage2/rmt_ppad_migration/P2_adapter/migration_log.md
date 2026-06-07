# P2 migration log

Phase: GCA-to-CLRHead channel adapter.

## 2026-05-25  P2 authored

Created `tools/gca_to_clr_adapter.py` per
appendix-path3-implementation-prompt.md sec 5.2:
- `GCAtoCLRAdapter(in_channels=256, out_channels=64, num_levels=3)`
- 3 independent `Conv2d(1x1, bias=False) -> BatchNorm2d -> ReLU` blocks
- Kaiming-normal init on convs, ones/zeros init on BN

Acceptance test (sec 5.3) implemented in `smoke_test()` plus two extras:
- backward-pass gradient check (all 3 projections receive non-zero grad)
- wrong-number-of-levels input raises ValueError

NB81 (`notebooks/stage2_notebook_81_P2_adapter.ipynb`) is a thin runner
that executes the smoke test on Colab. P2 is small enough that the entire
acceptance check finishes in a few seconds on CPU - no Drive I/O.

The adapter is a standalone module here for clean phase isolation. P4
will import it from this path and inject it into the vendored
MTDETRDecoder's `clr_lane` branch when wiring the LaneSegHead.

## Pending follow-ups

None yet.
