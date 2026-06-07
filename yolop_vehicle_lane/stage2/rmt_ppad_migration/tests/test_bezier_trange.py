"""Validate the bezier t-range fix rationale at the render-ops level.

The bezier head bug: reg_layers init gave t_start=0.5, t_end=0.55 -> each curve
rendered only ~5% of its span (a stub) -> near-zero mask overlap -> stuck IoU.
The fix biases t_start->~0.02, t_end->~0.98 (full span). This test confirms
render_with_validity actually honors the range: a near-full range must sample a
WIDE y-span, a 0.50..0.55 stub must sample a TINY y-span. Pure torch.
Results -> file (stdout unreliable in this env).
"""
import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MIG = HERE.parent
spec = importlib.util.spec_from_file_location(
    'bops', MIG / 'extensions/bezier_lcm/Bops/tools/bezier_ops.py')
bops = importlib.util.module_from_spec(spec); spec.loader.exec_module(bops)


def main():
    fails = []
    # control polygon: a roughly vertical lane from bottom(y=0.9) to top(y=0.1)
    # packed [P0x,P1x,P2x,P3x, P0y,P1y,P2y,P3y]
    cp8 = torch.tensor([[0.30, 0.37, 0.43, 0.50,
                         0.90, 0.63, 0.37, 0.10]])

    def yspan(t0, t1):
        xs, ys = bops.render_with_validity(
            cp8, torch.tensor([t0]), torch.tensor([t1]), n_samples=72, mode='cubic')
        ys = ys.detach().reshape(-1)
        return float(ys.max() - ys.min())

    full = yspan(0.02, 0.98)   # the FIX
    stub = yspan(0.50, 0.55)   # the BUG

    if full < 0.6:
        fails.append(f'full-range y-span={full:.3f} (<0.6 -> render ignores range/curve flat)')
    if stub > 0.15:
        fails.append(f'stub-range y-span={stub:.3f} (>0.15 -> range not honored)')
    if not (full > stub * 3):
        fails.append(f'full({full:.3f}) not >> stub({stub:.3f}) -> t-range has no effect')

    (HERE / 'bezier_trange_result.txt').write_text(
        (f'PASS full_span={full:.3f} stub_span={stub:.3f}' if not fails
         else 'FAIL\n' + '\n'.join(fails)) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
