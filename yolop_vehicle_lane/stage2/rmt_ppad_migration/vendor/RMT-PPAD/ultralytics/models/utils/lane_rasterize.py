"""Vendor-side shim re-exporting P7's lane_rasterize."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_p7():
    here = Path(__file__).resolve()
    cur = here.parent
    migration_root = None
    for _ in range(10):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            migration_root = cur
            break
        cur = cur.parent
    if migration_root is None:
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
        raise FileNotFoundError(f'lane_rasterize shim missed rmt_ppad_migration from {here}')
    p7 = migration_root / 'P7_validator' / 'tools' / 'lane_rasterize.py'
    if not p7.exists():
        raise FileNotFoundError(f'P7 lane_rasterize source missing at {p7}')
    spec = importlib.util.spec_from_file_location('p7_lane_rasterize', p7)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p7}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('p7_lane_rasterize', mod)
    spec.loader.exec_module(mod)
    return mod


_p7 = _import_p7()
lanes_to_mask = _p7.lanes_to_mask
batch_lanes_to_mask = _p7.batch_lanes_to_mask
lane_score_stats = _p7.lane_score_stats
__all__ = ['lanes_to_mask', 'batch_lanes_to_mask', 'lane_score_stats']
