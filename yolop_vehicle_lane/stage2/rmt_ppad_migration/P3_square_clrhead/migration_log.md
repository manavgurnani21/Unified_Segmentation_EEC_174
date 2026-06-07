# P3 migration log

Phase: CLRHead subclass with 192-prior layout retuned for 1:1 square
images (32 left + 128 bottom + 32 right).

## 2026-05-25  P3 authored

Created `tools/clr_head_square.py` per
appendix-path3-implementation-prompt.md sec 6.2-6.4:
- `CLRHeadForSquareImage(CLRHead)` overriding only `_init_prior_embeddings`
- Same total 192 priors (preserves the rest of the head's wiring)
- 32 left + 128 bottom + 32 right split (vs CULane's 24/144/24)

Created `tools/verify_priors.py` for the sec 6.5 acceptance test:
- Instantiates the head with a SimpleNamespace fake cfg
- Asserts (n_left, n_bottom, n_right) == (32, 128, 32)
- Asserts the generated priors tensor is shape (192, 78)
- Renders all 192 priors as colored arrows on a 640x640 PNG
  (green=left, cyan=bottom, red=right) for human inspection

## Import-chain shim for clrkd.ops.nms_impl

CLRKDNet's `clrkd/__init__.py` does `from .ops import *` which triggers
`from .nms import nms` which triggers `from . import nms_impl` (a CUDA
C++ extension). For P3 we only test prior initialization - the NMS op
is never called.

Rather than force everyone to compile the CUDA extension just to run a
CPU shape test, `clr_head_square._install_nms_stub_if_missing()` registers
a stub `clrkd.ops.nms_impl` module that raises a clear error if anyone
ever actually calls `nms_forward`. The shim runs at import time, BEFORE
`from clrkd.models.heads.clr_head import CLRHead`.

Real-inference paths (P7 validator, P8 training eval) will need the
real CUDA extension built. The stub raises a clear message pointing at
`python setup.py build_ext --inplace` in CLRKDNet-master/ so they get
unblocked quickly.

## NB82 acceptance

`notebooks/stage2_notebook_82_P3_square_clrhead.ipynb`:
- Cell 1: mount Drive, install mmcv (CLRHead's ConvModule dependency)
- Cell 2: run verify_priors.py via run_streaming
- Cell 3: display the priors_square.png inline + import the head
  in-process and confirm the per-region partition matches the spec

## 2026-05-25  Bug: mmcv 2.x removed mmcv.jit, CLRKDNet's accuracy.py uses it

NB82 first run failed with:
    AttributeError: module 'mmcv' has no attribute 'jit'
while loading `clrkd.models.losses.accuracy` (which CLRHead imports). Root
cause: that file decorates a function with `@mmcv.jit(coderize=True)` from
mmcv 1.x; mmcv 2.x (what Colab pip installs today, 2.2.0 in our session)
removed `mmcv.jit`.

Fix: `clr_head_square._install_mmcv_jit_shim_if_missing()` monkey-patches
`mmcv.jit` with a passthrough decorator before the CLRHead import. Handles
both `@mmcv.jit` and `@mmcv.jit(coderize=True)` call shapes. Same pattern
as the nms_impl stub - we never call mmcv.jit at runtime, we just need
the import chain to clear.

## 2026-05-25  Bug: arrow renderer used wrong y-axis + wrong direction formula

NB82 second run produced a priors_square.png with cyan (bottom-edge priors)
clustered at the IMAGE TOP, green (left) covering the left edge with weird
zigzag, and no red (right) arrows visible. Two coordinate-system bugs in
the renderer (the prior init itself was correct - smoke test passed
(32, 128, 32) partition and (192, 78) shape).

Bug 1 (y-axis flip):
  CLRHead's start_y is measured from the IMAGE BOTTOM (start_y=0 = bottom).
  My renderer did start_y_px = sy_norm * img_size which treats it as a
  top-down image coord. So bottom priors with start_y=0 rendered at
  pixel_y=0 (= image TOP).
  Fix: start_y_px = (1.0 - sy_norm) * img_size

Bug 2 (direction formula):
  Used end_x_px = start_x + L * sin(theta*pi). sin is positive for all
  theta in (0, 1), so right-edge priors (start_x=640, theta in [0.6, 0.8])
  always pointed further right off the canvas - invisible.
  CLRHead's actual lane formula has slope dx/d(y_up) = H / tan(theta*pi).
  cot(theta*pi) is negative for theta > 0.5, so right-edge arrows now
  correctly point UP-LEFT into the canvas.
  Fix: dx_px = dy_norm * H / tan(theta*pi); dy_px = -dy_norm * H

After re-run, expected: cyan arrows along bottom edge fanning upward,
green arrows along left edge shooting up-right, red arrows along right
edge shooting up-left. All in the lower-and-side portions of the canvas
because the lanes "head up" only ~30% of the image height per arrow.

## Pending follow-ups

None for P3 itself. P4 will use this subclass plus the GCAtoCLRAdapter
from P2 inside a LaneSegHead wrapper, and will need to install the same
two shims (or import via clr_head_square which already installs them).
