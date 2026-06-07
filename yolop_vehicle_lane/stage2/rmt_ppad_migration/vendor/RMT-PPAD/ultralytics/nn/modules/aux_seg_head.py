"""Vendor-side shim exposing aux_seg/'s AuxiliaryDenseHeads under
ultralytics.nn.modules, mirroring lane_head.py. The real implementation
lives at rmt_ppad_migration/aux_seg/aux_segmentors.py; reverting the aux
feature = deleting this file + that folder.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_aux_segmentors():
    here = Path(__file__).resolve()
    cur = here.parent
    migration_root = None
    for _ in range(10):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            migration_root = cur
            break
        cur = cur.parent
    if migration_root is None:
        for ancestor in here.parents:
            cand = ancestor / 'stage2' / 'rmt_ppad_migration'
            if cand.exists():
                migration_root = cand
                break
    if migration_root is None:
        raise FileNotFoundError(f'aux_seg_head shim missed rmt_ppad_migration from {here}')
    src = migration_root / 'aux_seg' / 'aux_segmentors.py'
    if not src.exists():
        raise FileNotFoundError(f'aux_seg source missing at {src}')
    spec = importlib.util.spec_from_file_location('aux_segmentors', src)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {src}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('aux_segmentors', mod)
    spec.loader.exec_module(mod)
    return mod


_aux = _import_aux_segmentors()
AuxiliaryDenseHeads = _aux.AuxiliaryDenseHeads
AUX_CH_DRIVABLE = _aux.AUX_CH_DRIVABLE
AUX_CH_LANE = _aux.AUX_CH_LANE

__all__ = ['AuxiliaryDenseHeads', 'AUX_CH_DRIVABLE', 'AUX_CH_LANE']
