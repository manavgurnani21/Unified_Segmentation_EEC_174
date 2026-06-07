"""Vendor-side shim re-exporting P5's lane_losses module.

Implementation lives at
yolop_vehicle_lane/stage2/rmt_ppad_migration/P5_loss/tools/lane_losses.py
for phase-isolation; this shim lets `from ultralytics.models.utils.lane_losses
import ...` work from inside the vendored package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _import_p5_lane_losses():
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
        raise FileNotFoundError(
            f'lane_losses shim could not find rmt_ppad_migration/ from {here}'
        )
    p5 = migration_root / 'P5_loss' / 'tools' / 'lane_losses.py'
    if not p5.exists():
        raise FileNotFoundError(f'P5 lane_losses source missing at {p5}')
    spec = importlib.util.spec_from_file_location('p5_lane_losses', p5)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not build spec for {p5}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('p5_lane_losses', mod)
    spec.loader.exec_module(mod)
    return mod


_p5 = _import_p5_lane_losses()

line_iou = _p5.line_iou
liou_loss = _p5.liou_loss
lane_iou = _p5.lane_iou              # S2.A: angle-aware CLRerNet LaneIoU
lane_iou_loss = _p5.lane_iou_loss    # S2.A: drop-in for liou_loss
distance_cost = _p5.distance_cost
focal_cost = _p5.focal_cost
dynamic_k_assign = _p5.dynamic_k_assign
assign = _p5.assign
FocalLossForLane = _p5.FocalLossForLane

__all__ = [
    'line_iou', 'liou_loss', 'lane_iou', 'lane_iou_loss',
    'distance_cost', 'focal_cost', 'dynamic_k_assign', 'assign',
    'FocalLossForLane',
]
