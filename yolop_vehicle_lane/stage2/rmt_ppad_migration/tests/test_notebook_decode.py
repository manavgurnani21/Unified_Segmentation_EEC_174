"""Verify the inline decode_lanes/read_boxes in NB101 & NB102 inspection cells
match the canonical (already-tested) GT rasterizer. These cells COPY the decode
logic as strings, so they can silently drift from the source of truth. We exec
the cell's functions in a sandbox and compare on a known GT tensor + label.
Results -> file (stdout unreliable in this env).
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIG = HERE.parent
NBDIR = MIG / 'notebooks'

gt_ras = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location('gtr', MIG / 'P6_dataset/tools/lane_mask_from_target.py'))
importlib.util.spec_from_file_location('gtr', MIG / 'P6_dataset/tools/lane_mask_from_target.py').loader.exec_module(gt_ras)

N, NS, S = 78, 71, 640


def gt_tensor(a0, L, x0, x1):
    r = np.full(N, -1e5, np.float32); r[0] = 0; r[1] = 1; r[2] = a0; r[5] = float(L)
    s = int(round(a0 * NS)); r[6 + s:6 + s + L] = np.linspace(x0, x1, L)
    return r


def _extract_funcs(nb_path):
    """Grab the cell that defines decode_lanes and exec just the two helpers."""
    nb = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    src = None
    for c in nb['cells']:
        s = ''.join(c['source'])
        if c['cell_type'] == 'code' and 'def decode_lanes' in s and 'def read_boxes' in s:
            src = s; break
    if src is None:
        return None, None
    # isolate the two function defs (up to the first non-indented line after each)
    g = {'np': np}
    # exec only the def blocks to avoid running the plotting/IO body
    lines = src.splitlines()
    blocks, cur, grab = [], [], False
    for ln in lines:
        if ln.startswith('def decode_lanes') or ln.startswith('def read_boxes'):
            if cur: blocks.append('\n'.join(cur)); cur = []
            grab = True
        elif grab and ln and not ln[0].isspace() and not ln.startswith('def '):
            blocks.append('\n'.join(cur)); cur = []; grab = False
        if grab:
            cur.append(ln)
    if cur: blocks.append('\n'.join(cur))
    code = '\n'.join(b for b in blocks if b.strip())
    # the funcs reference S, NS, NP_, YS as globals -> provide them
    g.update({'S': S, 'NS': NS, 'NP_': 72,
              'YS': np.arange(S, -1, -S / NS)[:72]})
    exec(code, g)
    return g.get('decode_lanes'), g.get('read_boxes')


def main():
    fails = []
    gt = gt_tensor(0.0, 40, 200, 300)

    # canonical decode: rows where GT mask is set, via the proven rasterizer
    canon = gt_ras.rasterize_lane_target_to_mask(
        np.stack([gt] + [np.full(N, -1e5, np.float32)] * 7), img_h=S, img_w=S, thickness=2)
    canon_rows = np.where(canon.any(axis=1))[0]
    canon_span = (canon_rows.min(), canon_rows.max())

    for nb in ('stage2_notebook_101_combined_winner_full_training.ipynb',
               'stage2_notebook_102_build_complete_dataset.ipynb'):
        dl, rb = _extract_funcs(NBDIR / nb)
        tag = nb.split('_')[2]  # 101 / 102
        if dl is None:
            fails.append(f'{tag}: could not extract decode_lanes'); continue
        lanes = dl(np.stack([gt]))
        if len(lanes) != 1:
            fails.append(f'{tag}: decode_lanes returned {len(lanes)} lanes (want 1)'); continue
        pts = lanes[0]
        ys = pts[:, 1]
        # the decoded polyline must span the SAME rows as the canonical mask (+-6px)
        if abs(int(ys.min()) - canon_span[0]) > 6 or abs(int(ys.max()) - canon_span[1]) > 6:
            fails.append(f'{tag}: decode y-span {int(ys.min())}..{int(ys.max())} '
                         f'!= canonical {canon_span}')
        # x near 200..300 (the encoded line)
        xs = pts[:, 0]
        if not (180 <= xs.min() <= 320 and 180 <= xs.max() <= 320):
            fails.append(f'{tag}: decode x-range {int(xs.min())}..{int(xs.max())} off (want ~200..300)')
        # read_boxes: YOLO "0 0.5 0.5 0.25 0.5" -> center 320,320 box
        if rb is not None:
            tmp = HERE / f'_lbl_{tag}.txt'; tmp.write_text('0 0.5 0.5 0.25 0.5\n')
            bxs = rb(tmp)
            tmp.unlink()
            if len(bxs) != 1:
                fails.append(f'{tag}: read_boxes got {len(bxs)} (want 1)')
            else:
                x1, y1, x2, y2 = bxs[0]
                if not (abs(x1 - 240) <= 2 and abs(x2 - 400) <= 2 and abs(y1 - 160) <= 2 and abs(y2 - 480) <= 2):
                    fails.append(f'{tag}: read_boxes coords {bxs[0]} != ~(240,160,400,480)')

    (HERE / 'notebook_decode_result.txt').write_text(
        ('PASS NB101+NB102 inline decode matches canonical' if not fails
         else 'FAIL\n' + '\n'.join(fails)) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
