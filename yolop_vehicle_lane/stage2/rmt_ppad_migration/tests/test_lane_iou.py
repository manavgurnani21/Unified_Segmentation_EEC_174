"""Validate our ported angle-aware LaneIoU (S2.A) against CLRerNet's own impl.

Risk register: "CLRerNet LaneIoU port has subtle frame/normalization bugs ->
unit-test against repo values on toy lanes before training." This does exactly
that: it runs BOTH our lane_iou (pixel-coord) and CLRerNet's LaneIoULoss
(relative-coord) on the same toy lanes and asserts they agree, plus the core
property that distinguishes LaneIoU from line_iou (steeper lanes get a wider
band -> a fixed x-error costs LESS IoU on a steep lane than line_iou would).

Result -> file (stdout unreliable in this sandbox).
"""
import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MIG = HERE.parent


def _load_ours():
    spec = importlib.util.spec_from_file_location(
        'll', MIG / 'P5_loss/tools/lane_losses.py')
    m = importlib.util.module_from_spec(spec)
    # lane_losses imports clrkd lazily only inside assign(); line/lane_iou are
    # pure torch, so exec is safe here.
    spec.loader.exec_module(m)
    return m


def _clrernet_laneiou(pred_rel, target_rel, base_rel, img_h, img_w):
    """Reference: a faithful inline copy of CLRerNet LaneIoULoss.calc_iou +
    _calc_lane_width (libs/models/losses/iou_loss.py), relative coords."""
    n_strips = pred_rel.shape[1] - 1
    dy = img_h / n_strips * 2

    def _w(x):
        dx = (x[:, 2:] - x[:, :-2]) * img_w
        w = base_rel * torch.sqrt(dx.pow(2) + dy ** 2) / dy
        return torch.cat([w[:, :1], w, w[:, -1:]], dim=1)
    pw = _w(pred_rel.clone().detach())
    tw = _w(target_rel)
    px1, px2 = pred_rel - pw, pred_rel + pw
    tx1, tx2 = target_rel - tw, target_rel + tw
    ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)
    bad = (target_rel < 0) | (target_rel >= 1.0)
    ovr[bad] = 0.0
    union[bad] = 0.0
    return ovr.sum(-1) / (union.sum(-1) + 1e-9)


def main():
    fails = []
    m = _load_ours()
    S = 640
    # toy: a gently-sloping lane and a steep one, plus a small x-offset version.
    base = torch.linspace(0.3, 0.5, 10).reshape(1, -1)        # gentle slope
    steep = torch.linspace(0.2, 0.8, 10).reshape(1, -1)       # steep slope
    off = 0.02                                                # 2%-of-width x-offset

    # 1. our lane_iou (pixels) must equal CLRerNet's LaneIoU (relative) for the
    #    SAME geometry + width. base_width: ours is in px (length), ref in
    #    relative; choose length=L px and base_rel=L/img_w so they match.
    L = 30
    for name, lane in (('gentle', base), ('steep', steep)):
        pred_px = (lane + off) * S
        tgt_px = lane * S
        ours = m.lane_iou(pred_px, tgt_px, S, length=L, aligned=True, img_h=S)
        ref = _clrernet_laneiou(lane + off, lane, L / S, S, S)
        if not torch.allclose(ours, ref, atol=2e-3):
            fails.append(f'{name}: ours={ours.item():.4f} != CLRerNet ref={ref.item():.4f}')

    # 2. the defining property: for the SAME absolute x-offset, the STEEP lane
    #    keeps MORE IoU under LaneIoU than under fixed-band line_iou, because its
    #    band widens. (line_iou would over-penalize the steep near-field.)
    pred_steep = (steep + off) * S
    tgt_steep = steep * S
    laneiou_steep = m.lane_iou(pred_steep, tgt_steep, S, length=L, aligned=True, img_h=S).item()
    lineiou_steep = m.line_iou(pred_steep, tgt_steep, S, length=L, aligned=True).item()
    if not (laneiou_steep > lineiou_steep):
        fails.append(f'steep: LaneIoU {laneiou_steep:.4f} not > line_iou '
                     f'{lineiou_steep:.4f} (angle-aware band should help steep lanes)')

    # 3. shape contract: pairwise mode returns (P, T)
    P = torch.cat([base, steep], 0) * S
    T = torch.cat([base, steep, base] , 0) * S
    mat = m.lane_iou(P, T, S, length=L, aligned=False, img_h=S)
    if tuple(mat.shape) != (2, 3):
        fails.append(f'pairwise shape {tuple(mat.shape)} != (2,3)')
    # diagonal-ish: pred row 0 (gentle) best-matches target col 0/2 (gentle copies)
    if not (mat[0, 0] > mat[0, 1]):
        fails.append('pairwise: gentle pred not closer to gentle target')

    # 4. loss is in [0,1] and lower for a closer pred
    near = m.lane_iou_loss((base + 0.005) * S, base * S, S, length=L, img_h=S).item()
    far = m.lane_iou_loss((base + 0.05) * S, base * S, S, length=L, img_h=S).item()
    if not (0 <= near <= 1 and near < far):
        fails.append(f'loss: near={near:.4f} far={far:.4f} (want 0<=near<far<=1)')

    (HERE / 'lane_iou_result.txt').write_text(
        (f'PASS LaneIoU matches CLRerNet (steep LaneIoU={laneiou_steep:.3f} > '
         f'line_iou={lineiou_steep:.3f}; loss near={near:.3f}<far={far:.3f})'
         if not fails else 'FAIL\n' + '\n'.join(fails)) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
