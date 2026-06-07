"""Auxiliary dense segmentation heads (training-only).

Re-introduces the original RMT-PPAD drivable-area + lane dense segmentation
supervision as AUXILIARY branches whose only job is to pump large, spatially
dense gradients into the SHARED backbone during training - countering the
"detection gradient domination" that was erasing road/lane features from the
trunk and collapsing the sparse CLR polyline head after ~epoch 5.

They are strictly training-only: at eval the decoder never constructs or
runs them, so inference is the Detection + CLR-polyline path with zero
auxiliary overhead (verified by tools/test_aux_routing.py).
"""
from __future__ import annotations

from .aux_segmentors import AuxiliaryDenseHeads, AUX_CH_DRIVABLE, AUX_CH_LANE

__all__ = ["AuxiliaryDenseHeads", "AUX_CH_DRIVABLE", "AUX_CH_LANE"]
