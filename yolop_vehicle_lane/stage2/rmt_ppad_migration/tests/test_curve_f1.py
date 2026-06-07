"""Permanent regression test for the curve-F1 metric (numpy/cv2 only).

The metric was fixed twice this session amid tool-layer confusion; this pins
the two root-cause fixes so they can't silently regress:
  A. broad curveIoU pool (top-64) so geometry is tracked independent of the
     near-flat early cls ranking  (NOT just the top-8 used for F1)
  B. predicted lanes are NOT dropped by row[1]<0.5 (that's a cls LOGIT for
     preds, a 0/1 flag only for GT)
Writes PASS/FAIL lines to result file (stdout is unreliable in this env).
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIG = HERE.parent
spec = importlib.util.spec_from_file_location(
    'lcf1', MIG / 'P7_validator' / 'tools' / 'lane_curve_f1.py')
lcf1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lcf1)

N, NS = 78, 71


def gt_row(a0, L, x0, x1):
    r = np.full(N, -1e5); r[0] = 0; r[1] = 1; r[2] = a0; r[5] = float(L)
    s = int(round(a0 * NS)); r[6 + s:6 + s + L] = np.linspace(x0, x1, L)
    return r


def pred_row(a0, L, x0, x1, pos):
    r = np.full(N, -1e5); r[0] = 0; r[1] = pos; r[2] = a0; r[5] = L / NS
    s = int(round(a0 * NS)); r[6 + s:6 + s + L] = np.linspace(x0, x1, L) / 639.0
    return r


def main():
    fails = []

    # 1. perfect match -> F1 = 1.0, curveIoU high
    gt = [gt_row(0.0, 40, 200, 300)]
    pr = [pred_row(0.0, 40, 200, 300, 6.0)]
    tp, fp, fn, isum, ng = lcf1.lane_curve_tp_fp_fn(np.array(pr), np.array(gt))
    f1 = lcf1.f1_from_counts(tp, fp, fn)['f1']
    if not (tp == 1 and abs(f1 - 1.0) < 1e-9):
        fails.append(f'perfect-match F1={f1} tp={tp} (want 1.0/1)')
    if (isum / max(1, ng)) < 0.9:
        fails.append(f'perfect-match curveIoU={isum/max(1,ng):.3f} (want >0.9)')

    # 2. FIX-B: a low-logit (pos<0) pred that geometrically matches must still
    #    be decoded for curveIoU (regression guard on the row[1] gating bug).
    gt2 = [gt_row(0.0, 40, 420, 440)]
    rows = [pred_row(0.4, 11, 85, 95, 9.0) for _ in range(8)]      # high-score stubs (miss)
    rows += [pred_row(0.0, 40, 422, 442, -3.0) for _ in range(4)]  # low-score match
    a = lcf1.lane_curve_tp_fp_fn(np.array(rows), np.array(gt2), curveiou_pool=8)
    b = lcf1.lane_curve_tp_fp_fn(np.array(rows), np.array(gt2))    # default pool=64
    ci8 = a[3] / max(1, a[4]); ci64 = b[3] / max(1, b[4])
    if ci8 >= 0.5:
        fails.append(f'pool=8 curveIoU={ci8:.3f} (want ~0, top-8 are stubs)')
    if ci64 < 0.5:
        fails.append(f'pool=64 curveIoU={ci64:.3f} (want >0.5 via FIX-A+B)')

    # 3. realistic 192-prior: 4 good among junk -> 4 TP
    gts = [gt_row(0.0, 40, 150, 200), gt_row(0.0, 40, 300, 320),
           gt_row(0.1, 35, 450, 480), gt_row(0.05, 38, 560, 600)]
    good = [(0.0, 40, 158, 208), (0.0, 40, 305, 325),
            (0.1, 35, 458, 488), (0.05, 38, 552, 592)]
    pr3 = [pred_row(*g, 8.0) for g in good]
    rng = np.random.default_rng(0)
    for _ in range(188):
        r = np.full(N, -1e5); r[0] = 2; r[1] = -2; r[2] = 0; r[5] = 0.4
        r[6:6 + 40] = rng.uniform(0, 1, 40); pr3.append(r)
    tp3, fp3, fn3, _, _ = lcf1.lane_curve_tp_fp_fn(np.array(pr3), np.array(gts))
    if tp3 != 4:
        fails.append(f'192-prior tp={tp3} (want 4)')

    # 4. empty GT -> all preds are FP, no crash
    tpe, fpe, fne, isume, nge = lcf1.lane_curve_tp_fp_fn(
        np.array(pr), np.zeros((0, N)))
    if not (tpe == 0 and nge == 0):
        fails.append(f'empty-GT tp={tpe} n_gt={nge} (want 0/0)')

    # 5. 16-D bezier decode path -> matched bezier gives a TP
    def bez(p0, p3, t0=0.02, t1=0.98, pos=6.0):
        r = np.zeros(16); r[0] = 0; r[1] = pos
        (x0, y0), (x3, y3) = p0, p3
        r[2:6] = [x0, (2 * x0 + x3) / 3, (x0 + 2 * x3) / 3, x3]
        r[6:10] = [y0, (2 * y0 + y3) / 3, (y0 + 2 * y3) / 3, y3]
        r[10] = t0; r[11] = t1
        return r
    bz = bez((0.30, 0.90), (0.50, 0.10))
    tpb, _, _, _, _ = lcf1.lane_curve_tp_fp_fn(np.array([bz]), np.array([bz]))
    if tpb != 1:
        fails.append(f'bezier matched tp={tpb} (want 1)')

    (HERE / 'curve_f1_result.txt').write_text(
        ('PASS all 5 curve-F1 checks' if not fails else
         'FAIL\n' + '\n'.join(fails)) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
