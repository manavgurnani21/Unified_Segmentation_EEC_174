"""Vendor-side shim re-exporting B4's lane_bezier_head module.

Implementation lives at
yolop_vehicle_lane/stage2/rmt_ppad_migration/extensions/bezier_lcm/
B4_head/tools/lane_bezier_head.py
for phase isolation; this shim lets
`from .lane_bezier_head import LaneBezierHead`
work from inside the vendored package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_b4():
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
        raise FileNotFoundError(
            f'lane_bezier_head shim could not find rmt_ppad_migration/ from {here}'
        )
    p = migration_root / 'extensions' / 'bezier_lcm' / 'B4_head' / 'tools' / 'lane_bezier_head.py'
    if not p.exists():
        raise FileNotFoundError(f'B4 lane_bezier_head source missing at {p}')
    spec = importlib.util.spec_from_file_location('b4_lane_bezier_head', p)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('b4_lane_bezier_head', mod)
    spec.loader.exec_module(mod)
    return mod


_b4 = _import_b4()
LaneBezierHead = _b4.LaneBezierHead
priors_to_bezier_init = _b4.priors_to_bezier_init

__all__ = ['LaneBezierHead', 'priors_to_bezier_init']
