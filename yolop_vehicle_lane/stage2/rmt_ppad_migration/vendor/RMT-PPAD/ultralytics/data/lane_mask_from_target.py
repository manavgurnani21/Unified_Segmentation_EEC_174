"""Vendor-side shim re-exporting P6's `rasterize_lane_target_to_mask`."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_p6():
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
        raise FileNotFoundError(f'lane_mask_from_target shim missed rmt_ppad_migration from {here}')
    p6 = migration_root / 'P6_dataset' / 'tools' / 'lane_mask_from_target.py'
    if not p6.exists():
        raise FileNotFoundError(f'P6 source missing at {p6}')
    spec = importlib.util.spec_from_file_location('p6_lane_mask_from_target', p6)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p6}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('p6_lane_mask_from_target', mod)
    spec.loader.exec_module(mod)
    return mod


_p6 = _import_p6()
rasterize_lane_target_to_mask = _p6.rasterize_lane_target_to_mask
__all__ = ['rasterize_lane_target_to_mask']
