"""Phase P3: CLRHead subclass with priors initialized for 1:1 (square) images.

CLRKDNet's original CLRHead._init_prior_embeddings places 24 left + 144 bottom
+ 24 right priors, sized for CULane's 1:2.5 aspect (320x800). For our square
640x640 BDD targets we re-balance to 32 left + 128 bottom + 32 right per
appendix-path3-implementation-prompt.md sec 6.3 - same 192 total so the rest
of the head's wiring is untouched.

Only `_init_prior_embeddings` is overridden. `generate_priors_from_embeddings`
and every other CLRHead method stays as-is.

Import note
-----------
CLRKDNet's `clrkd/__init__.py` triggers `from .ops import *` which triggers
`from . import nms_impl` (a CUDA C++ extension). For P3 we only need the
prior-init logic - we never run NMS. To avoid forcing every user to compile
the CUDA extension just to instantiate a head for a CPU shape test, this
module installs a tiny stub for `clrkd.ops.nms_impl` BEFORE importing
CLRHead. The stub raises if `nms_forward` is ever actually called, so
real-inference code paths still fail loudly.
"""
from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path
from typing import List, Optional


def _add_clrkd_to_syspath() -> Path:
    """Locate external_repos/CLRKDNet-master and add it to sys.path."""
    here = Path(__file__).resolve()
    # tools/ -> P3_square_clrhead/ -> rmt_ppad_migration/ -> stage2/ -> yolop_vehicle_lane/ -> REPO_ROOT
    repo_root = here.parent.parent.parent.parent.parent
    candidates = [
        repo_root / 'external_repos' / 'CLRKDNet-master',
        repo_root.parent / 'external_repos' / 'CLRKDNet-master',
    ]
    for c in candidates:
        if (c / 'clrkd' / '__init__.py').exists():
            cs = str(c)
            if cs not in sys.path:
                sys.path.insert(0, cs)
            return c
    raise FileNotFoundError(
        f'Could not locate CLRKDNet repo with clrkd/ package. Tried: '
        f'{[str(c) for c in candidates]}'
    )


def _install_nms_stub_if_missing() -> None:
    """Install a stub for clrkd.ops.nms_impl IF the real CUDA extension
    can't be imported. CLRHead only IMPORTS nms_impl at module load - it
    never CALLS it during __init__ or prior generation, so the stub is
    safe for P3's CPU-only shape tests."""
    try:
        importlib.import_module('clrkd.ops.nms_impl')
        return  # the real extension is built; nothing to do
    except (ImportError, OSError, Exception):  # noqa: BLE001
        pass

    def _nms_forward_stub(*args, **kwargs):
        raise RuntimeError(
            'clrkd.ops.nms_impl is a stub (the CUDA extension was not built). '
            'P3 prior-init tests do not call NMS, but production inference does. '
            'Build CLRKDNet with `python setup.py build_ext --inplace` from '
            'external_repos/CLRKDNet-master/.'
        )

    stub = types.ModuleType('clrkd.ops.nms_impl')
    stub.nms_forward = _nms_forward_stub
    sys.modules['clrkd.ops.nms_impl'] = stub


def _install_mmcv_jit_shim_if_missing() -> None:
    """CLRKDNet's accuracy.py uses `@mmcv.jit(coderize=True)` - that
    decorator existed in mmcv 1.x and was REMOVED in mmcv 2.x (which is
    what Colab pip-installs today). It was always optional JIT tracing,
    so we replace it with a passthrough decorator. Handles both forms:
        @mmcv.jit              -> the decorator IS the function call
        @mmcv.jit(coderize=True) -> the call returns a decorator

    Without this, `from clrkd.models.heads.clr_head import CLRHead`
    fails with `AttributeError: module 'mmcv' has no attribute 'jit'`
    while loading clrkd.models.losses.accuracy.
    """
    try:
        import mmcv  # noqa: WPS433
    except ImportError:
        return  # mmcv not installed at all -> let ConvModule import fail clearly
    if getattr(mmcv, 'jit', None) is not None:
        return

    def _jit_passthrough(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Bare @mmcv.jit
            return args[0]
        # Parameterized: @mmcv.jit(coderize=True) etc.
        def _decorator(fn):
            return fn
        return _decorator

    mmcv.jit = _jit_passthrough  # type: ignore[attr-defined]


_add_clrkd_to_syspath()
_install_nms_stub_if_missing()
_install_mmcv_jit_shim_if_missing()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from clrkd.models.heads.clr_head import CLRHead  # noqa: E402


class CLRHeadForSquareImage(CLRHead):
    """Re-initialize the 192 anchor priors for a 1:1 image aspect.

    Split (per appendix-path3 sec 6.3):
        32 left-side priors  (start_x = 0,   start_y varies, theta toward right)
        128 bottom priors    (start_x varies, start_y = 0,    theta near vertical)
        32 right-side priors (start_x = 1,   start_y varies, theta toward left)
    Total = 192.

    The base CLRHead's `_init_prior_embeddings` was tuned for CULane's 320x800
    crop where lanes overwhelmingly enter from the bottom. For square 640x640
    BDD frames, more lanes enter from the side (especially intersection scenes)
    so we move 16 priors out of the bottom group into the side groups.
    """

    def _init_prior_embeddings(self) -> None:
        # 2/3 of priors on the bottom edge, remaining 1/3 split evenly L/R.
        bottom_priors_nums = int(self.num_priors * 2 / 3)         # 128 of 192
        side_priors_nums = (self.num_priors - bottom_priors_nums) // 2  # 32 each
        left_priors_nums = side_priors_nums
        # right_priors_nums = self.num_priors - bottom_priors_nums - left_priors_nums (also 32)

        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        # ---- Left edge: start_x = 0, start_y in [0.1, 0.9], theta shooting right
        left_strip_size = (
            0.8 / max(1, (left_priors_nums // 2 - 1))
            if left_priors_nums > 2 else 0.4
        )
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * left_strip_size + 0.1)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.0)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.20 if i % 2 == 0 else 0.40)

        # ---- Bottom edge: start_y = 0, start_x in (0, 1), theta near-vertical (0.2..0.65)
        bottom_strip_size = 1.0 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
            local_idx = i - left_priors_nums
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.0)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              (local_idx // 4 + 1) * bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 + 0.15 * (i % 4))

        # ---- Right edge: start_x = 1, start_y in [0.1, 0.9], theta shooting left
        right_start = left_priors_nums + bottom_priors_nums
        right_count = self.num_priors - right_start
        right_strip_size = (
            0.8 / max(1, (right_count // 2 - 1))
            if right_count > 2 else 0.4
        )
        for i in range(right_start, self.num_priors):
            local_idx = i - right_start
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (local_idx // 2) * right_strip_size + 0.1)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.0)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.60 if i % 2 == 0 else 0.80)


__all__ = ['CLRHeadForSquareImage']
