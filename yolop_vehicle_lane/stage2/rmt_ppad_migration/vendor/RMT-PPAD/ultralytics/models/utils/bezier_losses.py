"""Vendor-side shim re-exporting B5's bezier_losses module.

Implementation lives at
yolop_vehicle_lane/stage2/rmt_ppad_migration/extensions/bezier_lcm/
B5_loss/tools/bezier_losses.py
for phase isolation; this shim lets
`from ultralytics.models.utils.bezier_losses import ...`
work from inside the vendored package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_b5():
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
            f'bezier_losses shim could not find rmt_ppad_migration/ from {here}'
        )
    p = migration_root / 'extensions' / 'bezier_lcm' / 'B5_loss' / 'tools' / 'bezier_losses.py'
    if not p.exists():
        raise FileNotFoundError(f'B5 bezier_losses source missing at {p}')
    spec = importlib.util.spec_from_file_location('b5_bezier_losses', p)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('b5_bezier_losses', mod)
    spec.loader.exec_module(mod)
    return mod


_b5 = _import_b5()
FocalLossForLane = _b5.FocalLossForLane
lane_bezier_geom_loss = _b5.lane_bezier_geom_loss
bezier_liou_loss = _b5.bezier_liou_loss
bezier_line_iou = _b5.bezier_line_iou
bezier_distance_cost = _b5.bezier_distance_cost
assign_bezier = _b5.assign_bezier
complexity_penalty_loss = _b5.complexity_penalty_loss

__all__ = [
    'FocalLossForLane', 'lane_bezier_geom_loss', 'bezier_liou_loss',
    'bezier_line_iou', 'bezier_distance_cost', 'assign_bezier',
    'complexity_penalty_loss',
]
