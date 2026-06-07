"""Vendor-side shim that exposes P4's LaneSegHead under the
ultralytics.nn.modules package, so head.py can do `from .lane_head import
LaneSegHead` and so the rest of the framework can `nn.Module.find('LaneSegHead')`.

The actual implementation lives at
yolop_vehicle_lane/stage2/rmt_ppad_migration/P4_lane_head_integration/tools/lane_seg_head.py
under phase-isolation. We import it dynamically by absolute path so the
vendored ultralytics package keeps no source-level coupling with our phase
folders - reverting P4 means deleting this single file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_p4_lane_seg_head():
    here = Path(__file__).resolve()
    # vendor/RMT-PPAD/ultralytics/nn/modules/lane_head.py
    # -> walk up to vendor/, then sideways to rmt_ppad_migration/P4_...
    cur = here.parent
    migration_root = None
    for _ in range(10):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            migration_root = cur
            break
        cur = cur.parent
    if migration_root is None:
        # Fallback: find any rmt_ppad_migration sibling via repo-root traversal.
        for _ in range(10):
            cand = here.parent
            for _ in range(10):
                if (cand / 'stage2' / 'rmt_ppad_migration').exists():
                    migration_root = cand / 'stage2' / 'rmt_ppad_migration'
                    break
                cand = cand.parent
            if migration_root is not None:
                break
    if migration_root is None:
        raise FileNotFoundError(
            f'lane_head.py shim could not locate rmt_ppad_migration/ from {here}'
        )
    p4 = migration_root / 'P4_lane_head_integration' / 'tools' / 'lane_seg_head.py'
    if not p4.exists():
        raise FileNotFoundError(f'P4 source missing at {p4}')
    spec = importlib.util.spec_from_file_location('p4_lane_seg_head', p4)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p4}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('p4_lane_seg_head', mod)
    spec.loader.exec_module(mod)
    return mod


_p4 = _import_p4_lane_seg_head()

LaneSegHead = _p4.LaneSegHead

__all__ = ['LaneSegHead']
