"""Vendor-side shim re-exporting P7's lane_curve_f1."""
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
        for ancestor in here.parents:
            cand = ancestor / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration'
            if cand.exists():
                migration_root = cand
                break
    if migration_root is None:
        raise FileNotFoundError(f'lane_curve_f1 shim missed rmt_ppad_migration from {here}')
    p7 = migration_root / 'P7_validator' / 'tools' / 'lane_curve_f1.py'
    if not p7.exists():
        raise FileNotFoundError(f'P7 lane_curve_f1 source missing at {p7}')
    spec = importlib.util.spec_from_file_location('p7_lane_curve_f1', p7)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p7}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('p7_lane_curve_f1', mod)
    spec.loader.exec_module(mod)
    return mod


_p7 = _import_p7()
lane_curve_tp_fp_fn = _p7.lane_curve_tp_fp_fn
f1_from_counts = _p7.f1_from_counts
__all__ = ['lane_curve_tp_fp_fn', 'f1_from_counts']
