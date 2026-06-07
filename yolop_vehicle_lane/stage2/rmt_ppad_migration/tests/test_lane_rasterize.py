"""Verify the lane rasterizer length-scaling fix + pred/GT encoding agreement.

This is the breakthrough fix (frozen IoU): PRED length is normalized and must be
scaled by n_strips; PRED x is normalized and scaled by (img_w-1). GT length is a
strip-count, GT x is pixels. A PRED and a GT encoding the SAME physical line must
rasterize to (nearly) the SAME mask. Results -> file (stdout unreliable here).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MIG = HERE.parent


def _imp(name, rel):
    spec = importlib.util.spec_from_file_location(name, MIG / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


lr = _imp('lr', 'P7_validator/tools/lane_rasterize.py')
gt_ras = _imp('gtr', 'P6_dataset/tools/lane_mask_from_target.py')

N, NS, S = 78, 71, 640


def gt_tensor(a0, L, x0, x1):
    r = np.full(N, -1e5, np.float32); r[0] = 0; r[1] = 1; r[2] = a0; r[5] = float(L)
    s = int(round(a0 * NS)); r[6 + s:6 + s + L] = np.linspace(x0, x1, L)
    return r


def pred_tensor(a0, L, x0, x1, pos=8.0):
    # PRED: length normalized (L/NS), x normalized (/639), pos is a logit
    r = np.full(N, -1e5, np.float32); r[0] = -pos; r[1] = pos; r[2] = a0
    r[5] = L / NS
    s = int(round(a0 * NS)); r[6 + s:6 + s + L] = np.linspace(x0, x1, L) / (S - 1)
    return r


def main():
    fails = []

    # GT mask via the proven GT rasterizer
    gt = np.stack([gt_tensor(0.0, 40, 200, 300)] + [np.full(N, -1e5, np.float32)] * 7)
    gt[1:, 0] = 1; gt[1:, 1] = 0   # rest invalid
    gt_mask = gt_ras.rasterize_lane_target_to_mask(gt, img_h=S, img_w=S, thickness=8)

    # PRED mask via the eval rasterizer (192 priors; 1 good + junk)
    pr = [pred_tensor(0.0, 40, 200, 300, pos=9.0)]
    rng = np.random.default_rng(0)
    for _ in range(191):
        r = np.full(N, -1e5, np.float32); r[0] = 2; r[1] = -2; r[2] = 0; r[5] = 0.4
        r[6:6 + 40] = rng.uniform(0, 1, 40); pr.append(r)
    pred_t = torch.tensor(np.stack(pr))
    pred_mask = lr.lanes_to_mask(pred_t, img_h=S, img_w=S, line_width=8,
                                 conf_threshold=0.0, max_lanes=8)

    gpx = int(gt_mask.sum()); ppx = int(pred_mask.sum())
    if gpx == 0:
        fails.append('GT mask empty (GT rasterizer broken)')
    if ppx == 0:
        fails.append('PRED mask empty -> length-scaling fix REGRESSED (frozen IoU)')

    # overlap: the matching pred lane should overlap the GT lane substantially
    inter = int(np.logical_and(gt_mask > 0, pred_mask > 0).sum())
    union = int(np.logical_or(gt_mask > 0, pred_mask > 0).sum())
    iou = inter / union if union else 0.0
    if iou < 0.3:
        fails.append(f'pred/GT IoU={iou:.3f} (<0.3 -> pred & GT encodings disagree)')

    # length-scaling regression guard: a pred with normalized length 0.56 (=40
    # strips) must draw ~40 rows, NOT round(0.56)=1 row (the old bug -> empty).
    only = torch.tensor(np.stack([pred_tensor(0.0, 40, 200, 300, 9.0)]
                                 + [pr[1]] * 7))
    m_one = lr.lanes_to_mask(only, img_h=S, img_w=S, line_width=2,
                             conf_threshold=0.0, max_lanes=1)
    rows = np.where(m_one.any(axis=1))[0]
    span = (rows.max() - rows.min()) if len(rows) else 0
    if span < 200:   # 40 strips over 640px ~ 360px of vertical span
        fails.append(f'length span only {span}px -> length not scaled by n_strips')

    (HERE / 'rasterize_result.txt').write_text(
        (f'PASS gt_px={gpx} pred_px={ppx} iou={iou:.3f} span={span}px'
         if not fails else 'FAIL\n' + '\n'.join(fails)) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
