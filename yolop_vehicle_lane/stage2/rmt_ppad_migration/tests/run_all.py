"""Run all local data-pipeline regression tests and write a summary.

These are CPU-only (numpy/cv2/torch) tests of the pure-Python data paths that
do NOT need the full clrkd/ultralytics stack or a GPU - so they run anywhere
and guard the bugs that actually bit us this project:
  - lane rasterizer length-scaling (the frozen-IoU breakthrough)
  - curve-F1 broad-pool + pred-row[1] gating (the curveIoU=0 fixes)
  - drivable extraction id-mask-vs-colormap + coverage (NB98's 4% bug)
  - notebook inline decode drift vs the canonical rasterizer

Usage:  python tests/run_all.py     (exit 0 = all pass; writes tests/RESULTS.txt)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ['test_curve_f1.py', 'test_lane_rasterize.py',
         'test_drivable_extract.py', 'test_notebook_decode.py',
         'test_bezier_trange.py', 'test_sprint1_levers.py', 'test_lane_iou.py']


def main():
    lines = []
    n_pass = 0
    for t in TESTS:
        rc = subprocess.run([sys.executable, str(HERE / t)],
                            capture_output=True, text=True).returncode
        status = 'PASS' if rc == 0 else f'FAIL(rc={rc})'
        if rc == 0:
            n_pass += 1
        lines.append(f'{status:12s} {t}')
    summary = f'{n_pass}/{len(TESTS)} passed'
    out = '\n'.join(lines) + f'\n\n{summary}\n'
    (HERE / 'RESULTS.txt').write_text(out)
    return 0 if n_pass == len(TESTS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
