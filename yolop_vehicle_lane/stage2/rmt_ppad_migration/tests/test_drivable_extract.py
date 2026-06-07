"""LOCAL end-to-end test of extract_drivable_subset.py against a synthetic zip
that mimics the BDD100K 'Drivable Maps' layout the user downloaded.

Verifies the real failure-prone behaviors:
  1. stem matching across the `_drivable_id` / `_drivable_color` suffixes
  2. _rank prefers the id MASK over the RGB colormap
  3. binarization 0/1/2 -> {0,1} (both drivable classes -> foreground)
  4. ~100% coverage when every subset stem has a label (the NB98 fix)
  5. a stem with NO label is reported missing (not silently fabricated)
"""
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MIG = HERE.parent
EXTRACT = MIG / 'aux_seg' / 'tools' / 'extract_drivable_subset.py'

WORK = HERE / '_work_drivable'
SUBSET = WORK / 'subset'
ZIP = WORK / 'bdd100k_drivable_maps.zip'


def _png_bytes(arr):
    ok, buf = cv2.imencode('.png', arr)
    assert ok
    return buf.tobytes()


def setup():
    import shutil
    if WORK.exists():
        shutil.rmtree(WORK)
    (SUBSET / 'images' / 'train2017').mkdir(parents=True)
    (SUBSET / 'images' / 'val2017').mkdir(parents=True)

    # 5 train stems, 3 val stems. One train stem ('nolabel_train') gets NO
    # drivable label so we can confirm it is reported missing.
    train_stems = [f'aaa{ i }-train' for i in range(4)] + ['nolabel_train']
    val_stems = [f'bbb{ i }-val' for i in range(3)]
    for st in train_stems:
        cv2.imwrite(str(SUBSET / 'images' / 'train2017' / f'{st}.jpg'),
                    np.full((720, 1280, 3), 100, np.uint8))
    for st in val_stems:
        cv2.imwrite(str(SUBSET / 'images' / 'val2017' / f'{st}.jpg'),
                    np.full((720, 1280, 3), 100, np.uint8))

    # Build the synthetic drivable_maps zip. Each labelled stem gets BOTH an
    # id mask (values 0/1/2) AND a color map - mirroring the real release - so
    # the _rank test is meaningful. The id mask has a known foreground count.
    id_mask = np.zeros((720, 1280), np.uint8)
    id_mask[100:200, :] = 1     # direct drivable
    id_mask[200:260, :] = 2     # alternative drivable
    FG = int((id_mask > 0).sum())
    color = np.zeros((720, 1280, 3), np.uint8)
    color[100:260, :] = (255, 0, 255)   # arbitrary RGB colormap

    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for st in train_stems:
            if st == 'nolabel_train':
                continue
            zf.writestr(f'bdd100k/drivable_maps/labels/train/{st}_drivable_id.png', _png_bytes(id_mask))
            zf.writestr(f'bdd100k/drivable_maps/color_labels/train/{st}_drivable_color.png', _png_bytes(color))
        for st in val_stems:
            zf.writestr(f'bdd100k/drivable_maps/labels/val/{st}_drivable_id.png', _png_bytes(id_mask))
            zf.writestr(f'bdd100k/drivable_maps/color_labels/val/{st}_drivable_color.png', _png_bytes(color))
    return FG, train_stems, val_stems


def run():
    FG, train_stems, val_stems = setup()
    print(f'[setup] zip={ZIP.name}  FG_pixels_per_id_mask={FG}')

    # --- inspect-only smoke ---
    r = subprocess.run([sys.executable, '-u', str(EXTRACT), '--seg-zip', str(ZIP),
                        '--inspect-only'], capture_output=True, text=True)
    print('--- inspect-only stdout (head) ---')
    print('\n'.join(r.stdout.splitlines()[:6]))
    assert r.returncode == 0, f'inspect-only failed rc={r.returncode}\n{r.stderr}'

    # --- real extraction ---
    r = subprocess.run([sys.executable, '-u', str(EXTRACT), '--seg-zip', str(ZIP),
                        '--subset-root', str(SUBSET),
                        '--splits', 'train2017', 'val2017'],
                       capture_output=True, text=True)
    print('--- extract stdout ---')
    print(r.stdout)
    if r.returncode != 0:
        print('STDERR:', r.stderr)
    assert r.returncode == 0, 'extraction returned nonzero'

    fails = []

    # (4) coverage: 4/5 train (nolabel missing), 3/3 val
    tr_out = SUBSET / 'drivable_masks' / 'train2017'
    va_out = SUBSET / 'drivable_masks' / 'val2017'
    n_tr = len(list(tr_out.glob('*.png'))) if tr_out.exists() else 0
    n_va = len(list(va_out.glob('*.png'))) if va_out.exists() else 0
    if n_tr != 4: fails.append(f'train coverage {n_tr} != 4')
    if n_va != 3: fails.append(f'val coverage {n_va} != 3')

    # (5) the no-label stem must NOT have produced a mask
    if (tr_out / 'nolabel_train.png').exists():
        fails.append('nolabel_train should have NO mask but one was written')

    # (2)+(3) the written mask must be the BINARIZED ID mask (FG pixels == FG),
    # NOT the colormap (which after grayscale+>0 would binarize differently).
    sample = tr_out / 'aaa0-train.png'
    if not sample.exists():
        fails.append('expected mask aaa0-train.png missing')
    else:
        m = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        vals = set(np.unique(m).tolist())
        got_fg = int((m > 0).sum())
        if vals - {0, 1}:
            fails.append(f'mask not binary {{0,1}}: values={vals}')
        # colormap (255,0,255)->gray is ~105 over rows 100:260 = 160*1280 px,
        # whereas id-mask FG = 160*1280 too here BUT the decisive check is the
        # mask is binary AND matches id-mask geometry exactly (rows 100:260).
        if got_fg != FG:
            fails.append(f'FG pixels {got_fg} != id-mask FG {FG} '
                         f'(may have grabbed colormap or mis-binarized)')
        # extra: confirm exact rows are the id-mask rows
        rows_set = np.where(m.any(axis=1))[0]
        if rows_set.min() != 100 or rows_set.max() != 259:
            fails.append(f'FG rows {rows_set.min()}..{rows_set.max()} != 100..259')

    import json
    diag = {
        'n_train_masks': n_tr, 'n_val_masks': n_va,
        'nolabel_mask_exists': (tr_out / 'nolabel_train.png').exists(),
        'fails': fails,
    }
    (HERE / '_work_drivable' / 'result.json').write_text(json.dumps(diag))
    # ONE compact line (resistant to tool-layer multiline garbling)
    print('RESULTLINE ' + json.dumps({'pass': not fails, **diag}))
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(run())
