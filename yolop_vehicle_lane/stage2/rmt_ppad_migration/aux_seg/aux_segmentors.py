"""AuxiliaryDenseHeads — training-only drivable + lane dense seg branch.

WHY (the problem): the dense detection head dominates the shared backbone's
gradients (negative transfer), erasing the road/spatial features the sparse
CLR polyline lane head depends on; lane quality peaks ~ep5 then erodes.

WHAT (the fix): re-attach a dense semantic-segmentation branch to the shared
FPN features ONLY during training. Its big, spatially dense gradients push
the backbone to keep road/drivable/lane structure, fighting the detection
gradient. At inference the branch is bypassed entirely (see the decoder's
`if self.training and self.use_aux_seg` gate), so it adds zero test-time cost
and is droppable.

DESIGN: rather than copy/fork the original RMT-PPAD decoder (which would
diverge over time), we REUSE the already-vendored `TransformerSegmentationDecoder`
— that class *is* the original RMT-PPAD multi-class seg decoder (one decoder,
N output channels; the legacy non-lane-only loss reads ch0=drivable, ch1=lane).
`AuxiliaryDenseHeads` is the thin, isolated wrapper that owns it, exposes the
two channels by name, and keeps all aux logic in this folder. Reverting the
whole feature = stop setting `USE_AUX_SEG` (and delete this folder + shim).

The output is `(B, num_classes, H, W)` logits. Channel convention matches the
legacy loss so the GT masks line up:
    channel 0 = drivable area
    channel 1 = lane
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn

AUX_CH_DRIVABLE = 0
AUX_CH_LANE = 1


def _load_native_seg_decoder():
    """Return the vendored original RMT-PPAD `TransformerSegmentationDecoder`.

    Prefer the normal package import (ultralytics is on sys.path during
    training); fall back to an absolute-path import so this module is also
    importable from a bare verification script.
    """
    try:
        from ultralytics.nn.modules.transformer import TransformerSegmentationDecoder
        return TransformerSegmentationDecoder
    except Exception:  # noqa: BLE001 - fall back to path import
        import importlib.util
        import sys
        here = Path(__file__).resolve()
        mig = None
        cur = here.parent
        for _ in range(10):
            if cur.name == "rmt_ppad_migration" and (cur / "README.md").exists():
                mig = cur
                break
            cur = cur.parent
        if mig is None:
            raise
        tr = mig / "vendor" / "RMT-PPAD" / "ultralytics" / "nn" / "modules" / "transformer.py"
        spec = importlib.util.spec_from_file_location("_aux_native_transformer", tr)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_aux_native_transformer", mod)
        spec.loader.exec_module(mod)
        return mod.TransformerSegmentationDecoder


class AuxiliaryDenseHeads(nn.Module):
    """Unified training-only auxiliary dense seg head (drivable + lane).

    Args:
        hidden_dim: channel dim of the shared FPN features fed in (256).
        num_classes: dense seg output channels. 2 = [drivable, lane]
            (matches BDD's `segmentation: [1, 2]`). Set 1 for lane-only aux.
        img_size: square image size the masks are upsampled to (640).
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2, img_size: int = 640):
        super().__init__()
        TransformerSegmentationDecoder = _load_native_seg_decoder()
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        # The original RMT-PPAD dense decoder: (list[level feats], imgsz) ->
        # (seg_mask (B, num_classes, H, W), aux_list).
        self.decoder = TransformerSegmentationDecoder(hidden_dim, self.num_classes)

    def forward(self, x_proj_seg):
        """x_proj_seg: list of per-level shared FPN feature maps (B,256,h,w).

        Returns the dense seg logits (B, num_classes, H, W). The decoder also
        returns an aux list we don't need here.
        """
        seg_mask, _aux = self.decoder(x_proj_seg, self.img_size)
        return seg_mask


__all__ = ["AuxiliaryDenseHeads", "AUX_CH_DRIVABLE", "AUX_CH_LANE"]
