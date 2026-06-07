"""Wrapper around CLRKDNet's official ``CLRHead`` (vendored at
``external_repos/CLRKDNet-master``).

This is the **Option C smoke-test** wrapper. It is intentionally minimal:

- Adds ``external_repos/CLRKDNet-master`` to ``sys.path`` so we can import
  ``clrkd.models.heads.clr_head.CLRHead`` directly without modifying the vendor
  tree (per the user's plan in ``appendix-path3-implementation-prompt.md``
  section 14: "Do not modify the CLRKDNet vendor source").
- Monkey-patches ``clrkd.ops.nms_impl`` to a Python stub so the import does
  NOT require the compiled CUDA extension. We only use ``nms`` during the
  ``get_lanes()`` inference path, which the smoke test does not exercise.
- Builds a tiny ``cfg`` shim (a ``SimpleNamespace`` with the keys CLRHead
  reads in its ``__init__`` and ``forward``).
- Adds 1x1 channel adapters that map our backbone's per-level feature channels
  (typically 128) to CLRHead's ``prior_feat_channels`` (default 64).
- Translates CLRHead's ``(B, num_priors, 78)`` output into the dict format our
  ``FusionLaneLoss`` and ``LaneF1DecodedMetric`` already consume
  (``cls_logits`` (B, P), ``coord_pred`` (B, P, N, 2)).

What this wrapper does NOT do (deferred until Option A or B is chosen):

- It does **not** call ``CLRHead``'s internal loss path. We always run the
  head in eval mode so we get raw predictions back; downstream loss code is
  responsible for supervision.
- It does **not** implement the aspect-ratio fix described in Phase P3 of the
  user's plan (re-initializing the 192 priors for square images). CLRHead's
  default prior init assumes 1:2.5 CULane aspect; on our 384x640 images the
  priors will be sub-optimally placed. This is acceptable for a smoke test
  (we just need the shapes to be correct); a real run would either reinit
  the priors or fine-tune the head long enough that the priors converge to
  better positions.
- It does **not** expose ``mask_logit`` since CLRHead's seg_decoder is only
  built/used in training mode (which we bypass).

If the smoke test passes, the next step is to decide between:
  Option A -- execute the full Path-3 plan (P0-P8) in the RMT-PPAD vendor tree;
  Option B -- adapt the plan to our ``yolop_vehicle_lane/stage2/`` codebase
              (add prior re-init, training-mode loss bypass, KD teacher mode).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import torch
import torch.nn as nn


# Resolve the CLRKDNet vendor root once.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLRKD_PATH = _REPO_ROOT / 'external_repos' / 'CLRKDNet-master'


def _ensure_clrkd_on_path() -> None:
    """Insert the CLRKDNet vendor directory at the front of ``sys.path`` so
    ``import clrkd.models.heads.clr_head`` resolves to it.
    """
    if not _CLRKD_PATH.exists():
        raise RuntimeError(
            f'CLRKDNet vendor directory not found at {_CLRKD_PATH}. '
            'Make sure external_repos/CLRKDNet-master is checked out.')
    p = str(_CLRKD_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)


def _install_mmcv_jit_shim() -> None:
    """Shim ``mmcv.jit``, which CLRKDNet uses as a JIT decorator on
    ``clrkd.models.losses.accuracy.accuracy``.

    ``mmcv.jit`` existed in mmcv 1.x as an optional TorchScript-compile
    decorator and was REMOVED in mmcv 2.x. Colab installs mmcv 2.x by
    default, so the bare CLRKDNet import chain hits
    ``AttributeError: module 'mmcv' has no attribute 'jit'`` inside
    ``accuracy.py`` (which is unconditionally imported by ``clr_head.py``
    even though our smoke test never calls the accuracy metric).
    Shimming to a no-op decorator lets the import chain complete.

    Supports both call patterns CLRKDNet uses:
      ``@mmcv.jit``               -- ``args = (fn,)``, returns ``fn``
      ``@mmcv.jit(coderize=True)`` -- ``args = ()``, returns a decorator.
    """
    try:
        import mmcv  # noqa: WPS433  (local import so the module loads lazily)
    except ImportError:
        # If mmcv isn't even installed, the CLRHead import will fail later
        # with a clearer message in import_clrhead(). Don't preempt it.
        return
    if hasattr(mmcv, 'jit'):
        return  # mmcv 1.x already provides it.

    def _jit_shim(*args, **kwargs):
        # @mmcv.jit (no parens) -> args=(fn,), kwargs={}
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # @mmcv.jit(coderize=True, ...) -> return a no-op decorator
        def _decorator(fn):
            return fn
        return _decorator

    mmcv.jit = _jit_shim  # type: ignore[attr-defined]


def _install_nms_stub() -> None:
    """Pre-register a stub ``clrkd.ops.nms_impl`` module so importing
    ``clrkd.ops`` (which does ``from . import nms_impl``) does not require
    the compiled CUDA extension.

    Calling the stub raises a clear error pointing at how to compile the real
    extension; our forward-pass smoke test never reaches it.
    """
    if 'clrkd.ops.nms_impl' in sys.modules:
        return

    stub = types.ModuleType('clrkd.ops.nms_impl')

    def _nms_forward_stub(*args, **kwargs):  # pragma: no cover -- error path
        raise RuntimeError(
            'clrkd.ops.nms_impl is a stub. Compile the real CUDA extension '
            'via `cd external_repos/CLRKDNet-master && python setup.py '
            'build develop` if you need real NMS for inference.')

    stub.nms_forward = _nms_forward_stub
    sys.modules['clrkd.ops.nms_impl'] = stub


def import_clrhead():
    """Import CLRKDNet's ``CLRHead`` class. Installs the path + stubs first.

    Raises ``ImportError`` with an actionable message if dependencies are
    missing (most commonly ``mmcv`` for ``mmcv.cnn.ConvModule``).
    """
    _ensure_clrkd_on_path()
    _install_mmcv_jit_shim()  # must run before any clrkd.* import
    _install_nms_stub()
    try:
        from clrkd.models.heads.clr_head import CLRHead  # type: ignore
    except (ImportError, AttributeError) as e:
        # AttributeError because mmcv 2.x raises AttributeError when CLRKDNet
        # uses APIs that no longer exist (e.g. mmcv.jit before our shim, or
        # other 1.x-only attributes we haven't shimmed yet).
        raise ImportError(
            f'Failed to import CLRKDNet CLRHead: {e}. '
            'Most likely cause: mmcv is not installed, or mmcv 2.x removed an '
            'API CLRKDNet expected. If the error mentions mmcv, try '
            '`pip install mmcv-full==1.7.2` (matches CLRKDNet\'s era) or, '
            'preferred, extend _install_mmcv_jit_shim() to cover the missing '
            'attribute. CLRHead also uses `mmcv.cnn.ConvModule` and CLRKDNet\'s '
            'registry pattern.') from e
    return CLRHead


class VendorCLRNetHead(nn.Module):
    """Smoke-test wrapper that lets our pipeline drive CLRKDNet's actual CLRHead.

    Input format: list/tuple of 3 feature tensors ``(B, C_i, H_i, W_i)`` from
    our backbone (P3, P4, P5 typically). The wrapper applies one 1x1 conv per
    level to project to ``prior_feat_channels`` (default 64), then runs
    CLRHead in eval mode to get raw predictions.

    Output format (compatible with ``FusionLaneLoss`` and
    ``LaneF1DecodedMetric``):

      ``cls_logits``: (B, num_priors)  - pos - neg logit
      ``coord_pred``: (B, num_priors, num_points, 2)  - (x, y) in [0, 1]
      ``lane_param``: (B, num_priors, 4)  - [start_y, start_x, theta, length]
      plus several optional keys our downstream code may inspect.

    Smoke-test only: training-mode loss is bypassed. See module docstring
    for full caveats.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        prior_feat_channels: int = 64,
        num_priors: int = 192,
        num_points: int = 72,
        sample_points: int = 36,
        refine_layers: int = 3,
        fc_hidden_dim: int = 64,
        num_fc: int = 2,
        img_h: int = 384,
        img_w: int = 640,
        num_classes: int = 2,
        bg_weight: float = 0.4,
        ignore_label: int = 255,
        # For back-compat with our pipeline's introspection -- not used by CLRHead.
        max_lanes: int = 10,
        num_lane_classes: int = 7,
        mask_size: Sequence[int] = (72, 128),
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(
                f'VendorCLRNetHead expects 3 input feature levels; got '
                f'{len(in_channels)} ({in_channels})')
        self.in_channels = [int(c) for c in in_channels]
        self.prior_feat_channels = int(prior_feat_channels)
        self.refine_layers = int(refine_layers)
        self.num_priors = int(num_priors)
        self.num_points = int(num_points)
        self.fc_hidden_dim = int(fc_hidden_dim)

        # 1x1 channel adapters: our backbone -> CLRHead's expected channels.
        self.adapters = nn.ModuleList([
            nn.Conv2d(c, self.prior_feat_channels, kernel_size=1, bias=False)
            for c in self.in_channels
        ])
        for adapter in self.adapters:
            nn.init.kaiming_normal_(adapter.weight, mode='fan_out',
                                    nonlinearity='relu')

        # CLRHead reads these attrs in __init__. ``haskey`` is a method CLRHead
        # calls in its loss path (which we never run here) to check optional
        # weight overrides; return False so defaults are used.
        cfg = SimpleNamespace(
            img_w=int(img_w),
            img_h=int(img_h),
            num_classes=int(num_classes),
            bg_weight=float(bg_weight),
            ignore_label=int(ignore_label),
        )
        cfg.haskey = lambda key: False

        # Import and instantiate the real CLRHead.
        CLRHead = import_clrhead()
        self.head = CLRHead(
            num_points=self.num_points,
            prior_feat_channels=self.prior_feat_channels,
            fc_hidden_dim=self.fc_hidden_dim,
            num_priors=self.num_priors,
            num_fc=int(num_fc),
            refine_layers=self.refine_layers,
            sample_points=int(sample_points),
            cfg=cfg,
        )

        # Back-compat attributes our other code may inspect.
        self.max_lanes = int(max_lanes)
        self.num_lane_classes = int(num_lane_classes)
        self.embed_dim = self.fc_hidden_dim
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        # FusionLaneLoss.match_targets reads .num_priors directly off the head.
        # Already set above; this is just documentation.

    @torch.no_grad()  # for the smoke-test path; remove decorator when training
    def _forward_eval(self, adapted: List[torch.Tensor]) -> torch.Tensor:
        """Run CLRHead in eval mode and return its final-stage predictions
        tensor of shape ``(B, num_priors, 78)``.
        """
        was_training = self.head.training
        self.head.eval()
        try:
            preds = self.head(adapted)
        finally:
            self.head.train(was_training)
        return preds

    def forward(self, feats):
        """Wrapper forward.

        Args:
            feats: list/tuple of 3 tensors ``(B, C_i, H_i, W_i)``.

        Returns:
            dict with keys ``cls_logits``, ``coord_pred``, ``lane_param``,
            ``lane_offsets``, ``lane_class_logits``, ``selected_prior_indices``,
            ``prior_select_logits``, ``aux_stage_outputs``, ``mask_logit``,
            and ``_clrnet_raw_preds`` (the raw 78-D tensor for debugging).
        """
        if not isinstance(feats, (list, tuple)):
            raise TypeError(f'feats must be list/tuple; got {type(feats)}')
        if len(feats) != 3:
            raise ValueError(
                f'VendorCLRNetHead expects 3 feature maps; got {len(feats)}')

        # Adapt channel counts.
        adapted = [self.adapters[i](feats[i]) for i in range(3)]

        # Run CLRHead (eval mode -- bypasses internal loss).
        preds = self._forward_eval(adapted)
        B, P, D = preds.shape
        assert P == self.num_priors, (
            f'CLRHead returned {P} priors; expected {self.num_priors}')
        assert D == 6 + self.num_points, (
            f'CLRHead returned {D}-D vectors; expected {6 + self.num_points} '
            f'(2 scores + 4 params + {self.num_points} x-offsets)')

        # Translate the 78-D format to our pipeline's dict.
        # CLRHead's layout: [neg_score, pos_score, start_y, start_x, theta,
        # length, x_0..x_71]; x's are normalized into [0, 1] image x.
        cls_logits = preds[:, :, 1] - preds[:, :, 0]   # (B, P)
        xs = preds[:, :, 6:]                            # (B, P, N) in [0, 1]
        # CLRHead's prior_ys = linspace(1, 0, N), so y[i] is shared across all
        # priors and decreases from 1 (bottom) to 0 (top).
        ys = torch.linspace(1.0, 0.0, self.num_points,
                            device=preds.device, dtype=preds.dtype)
        ys = ys.view(1, 1, self.num_points).expand(B, P, self.num_points)
        coord_pred = torch.stack([xs, ys], dim=-1).clamp(0.0, 1.0)

        out = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_param': preds[:, :, 2:6],
            'lane_offsets': torch.zeros_like(xs),
            'lane_class_logits': torch.zeros(
                B, P, self.num_lane_classes, device=preds.device, dtype=preds.dtype),
            'selected_prior': preds[:, :, 2:5],
            'selected_prior_indices': torch.arange(
                P, device=preds.device).view(1, -1).expand(B, -1),
            'prior_select_logits': cls_logits,
            'aux_stage_outputs': [],
            'mask_logit': None,
            '_clrnet_raw_preds': preds,
        }
        return out
