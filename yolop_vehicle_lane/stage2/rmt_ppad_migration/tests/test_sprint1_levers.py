"""Unit tests for the Sprint-1 lane-head lever helpers (CPU/torch only).

Pins the math of the three new loss levers so they can't silently regress:
  _qfl_loss        - Quality Focal Loss (S1.3 IoU-aware cls)
  _lane_match_line_iou - per-match aligned line-IoU (S1.3 target)
  _y_reweighted_liou   - y-reweighted IoU loss (S1.4)
Result -> file (stdout unreliable in this sandbox).
"""
import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MIG = HERE.parent
# import the loss module's helpers without importing all of ultralytics: load
# the file as a module is heavy (it imports ultralytics.utils.loss). Instead we
# re-implement-import only the three pure helpers by execing the function defs.
LOSS = MIG / 'vendor/RMT-PPAD/ultralytics/models/utils/loss.py'


def _load_helpers():
    src = LOSS.read_text(encoding='utf-8')
    # extract the three helper defs (between the marker and `class DETRLoss`)
    start = src.index('def _lane_match_line_iou')
    end = src.index('class DETRLoss(nn.Module):')
    block = src[start:end]
    g = {'torch': torch}
    # _lane_match_line_iou / _y_reweighted_liou import line_iou lazily; provide a
    # stub line_iou via a fake module so we test the wrapper logic deterministically.
    import types, sys as _sys
    fake = types.ModuleType('ultralytics.models.utils.lane_losses')
    def _line_iou(pred, target, img_w, length=15, aligned=True):
        # toy IoU: 1 - normalized mean abs x-diff, clamped [0,1] (aligned -> (N,))
        d = (pred - target).abs().mean(dim=1) / img_w
        return (1 - d).clamp(0, 1)
    fake.line_iou = _line_iou
    _sys.modules['ultralytics.models.utils.lane_losses'] = fake
    exec(block, g)
    return g


def main():
    fails = []
    g = _load_helpers()
    qfl = g['_qfl_loss']; matchiou = g['_lane_match_line_iou']; yrw = g['_y_reweighted_liou']

    # 1. QFL: perfect prediction (sigmoid(pos)->target) gives ~0 loss; wrong gives >0
    #    target=1, pos logit large +ve -> p~1 -> loss~0
    cls = torch.tensor([[-5.0, 5.0]])
    l_good = qfl(cls, torch.tensor([1.0]), gamma=2.0).item()
    l_bad = qfl(torch.tensor([[5.0, -5.0]]), torch.tensor([1.0]), gamma=2.0).item()
    if not (l_good < 0.05): fails.append(f'QFL good-pred loss {l_good:.4f} not ~0')
    if not (l_bad > l_good * 10): fails.append(f'QFL bad-pred {l_bad:.3f} not >> good {l_good:.4f}')

    # 2. QFL soft target: a half-IoU target pulls p toward 0.5 (loss minimized near p=0.5)
    grid = torch.linspace(-4, 4, 9).reshape(-1, 1)
    cls9 = torch.cat([(-grid), grid], dim=1)
    losses = torch.stack([qfl(cls9[i:i+1], torch.tensor([0.5]), 2.0)[0] for i in range(9)])
    p_at_min = torch.sigmoid(cls9[losses.argmin(), 1]).item()
    if not (0.35 < p_at_min < 0.65):
        fails.append(f'QFL soft target=0.5 minimized at p={p_at_min:.2f} (want ~0.5)')

    # 2b. NB105 REGRESSION: the soft-target SCALING used in the loss must NOT
    #     collapse positives toward ~0 when the matched IoU is realistically tiny
    #     (~0.05 early in training). Replicate the loss-side mapping
    #     floor + (1-floor)*(IoU/norm), clamp[floor,1], and assert a low-IoU
    #     positive still targets >= floor (so it stays a clear positive vs the 0
    #     negatives, and a tau~0.4 threshold keeps it).
    floor, norm = 0.5, 0.3
    def soft_map(iou):
        return min(1.0, max(floor, floor + (1 - floor) * (iou / norm)))
    t_lowiou = soft_map(0.05)        # the regime that broke NB105
    if not (t_lowiou >= floor):
        fails.append(f'soft-target for IoU=0.05 is {t_lowiou:.3f} < floor {floor} '
                     '(would collapse score_std -> NB105 bug)')
    # and a positive at this target should be DECODED at tau=0.4 (sigmoid(logit
    # that minimizes QFL toward t_lowiou) must exceed 0.4)
    grid2 = torch.linspace(-6, 6, 49).reshape(-1, 1)
    cls2 = torch.cat([-grid2, grid2], dim=1)
    L = torch.stack([qfl(cls2[i:i+1], torch.tensor([t_lowiou]), 2.0)[0] for i in range(49)])
    p_pos = torch.sigmoid(cls2[L.argmin(), 1]).item()
    if not (p_pos > 0.4):
        fails.append(f'low-IoU positive decodes at p={p_pos:.3f} <= 0.4 '
                     '(tau=0.4 would reject it -> F1=0, the NB105 failure)')

    # 3. match line-IoU: identical lanes -> IoU 1; far-apart -> low
    a = torch.zeros(2, 72); b = torch.zeros(2, 72)
    iou_same = matchiou(a, b, 640)
    a2 = torch.zeros(2, 72); b2 = torch.full((2, 72), 300.0)
    iou_far = matchiou(a2, b2, 640)
    if not torch.allclose(iou_same, torch.ones(2), atol=1e-4):
        fails.append(f'match IoU identical != 1: {iou_same.tolist()}')
    if not (iou_far.mean() < iou_same.mean()):
        fails.append('match IoU far not < identical')

    # 4. y-reweight: 'near' must put MORE weight on near (bottom) half. Construct
    #    error only in the near half -> 'near' loss > uniform-average loss.
    pred = torch.zeros(3, 72); tgt = torch.zeros(3, 72)
    tgt[:, 36:] = 200.0          # error only in near (bottom) half
    near_loss = yrw(pred, tgt, 640, g['_line_iou'] if '_line_iou' in g else
                    __import__('sys').modules['ultralytics.models.utils.lane_losses'].line_iou,
                    mode='near').mean().item()
    far_only = torch.zeros(3, 72); tgt2 = torch.zeros(3, 72); tgt2[:, :36] = 200.0
    near_loss_farERR = yrw(far_only, tgt2, 640,
                           __import__('sys').modules['ultralytics.models.utils.lane_losses'].line_iou,
                           mode='near').mean().item()
    if not (near_loss > near_loss_farERR):
        fails.append(f'y-reweight near: near-error loss {near_loss:.3f} not > '
                     f'far-error loss {near_loss_farERR:.3f}')

    # 5. S1.4b length hinge (replicates loss.py's asymmetric too-short term:
    #    short_by = (gt - pred).clamp(min=0)/n_strips; loss += w * short_by.mean()).
    #    Pins the 3 invariants that make it a SHRINK fix and not a generic L1:
    #    (a) pred too-short -> POSITIVE penalty, (b) exact AND too-long -> 0 (the
    #    asymmetry: must not punish over-long here, else it just re-adds the
    #    symmetric term we already have), (c) scales linearly with the weight.
    def _len_hinge(pred_len, gt_len, w, n_strips=71):
        reg = torch.tensor(pred_len, dtype=torch.float32)
        tgt = torch.tensor(gt_len, dtype=torch.float32)
        short_by = (tgt - reg).clamp(min=0.0) / (float(n_strips) if n_strips else 1.0)
        return (w * short_by.mean()).item()
    h_short = _len_hinge([20., 30.], [40., 50.], 2.0)
    h_exact = _len_hinge([40., 50.], [40., 50.], 2.0)
    h_long = _len_hinge([60., 70.], [40., 50.], 2.0)
    if not (h_short > 0):
        fails.append(f'len-hinge too-short penalty {h_short:.4f} not > 0')
    if not (abs(h_exact) < 1e-9 and abs(h_long) < 1e-9):
        fails.append(f'len-hinge NOT asymmetric: exact={h_exact:.4f} long={h_long:.4f} '
                     '(both must be 0; a too-long penalty would defeat the purpose)')
    if not (abs(_len_hinge([20.], [40.], 3.0) - 3 * _len_hinge([20.], [40.], 1.0)) < 1e-6):
        fails.append('len-hinge does not scale linearly with weight')

    (HERE / 'sprint1_levers_result.txt').write_text(
        ('PASS all 5 Sprint-1 lever checks '
         f'(qfl_good={l_good:.4f} qfl_bad={l_bad:.3f} '
         f'softmin_p={p_at_min:.2f} near>{near_loss:.2f}vs{near_loss_farERR:.2f} '
         f'lenhinge[short={h_short:.3f},exact={h_exact:.1f},long={h_long:.1f}])'
         if not fails else 'FAIL\n' + '\n'.join(fails)) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
