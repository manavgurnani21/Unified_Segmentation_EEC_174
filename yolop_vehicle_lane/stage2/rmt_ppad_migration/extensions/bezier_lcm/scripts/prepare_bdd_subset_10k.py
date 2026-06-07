"""Phase B (NB89): prepare a 10k-image stratified BDD subset.

Thin wrapper around `P8_train/scripts/prepare_bdd_subset.prepare()` that
just plugs in `n_train=10000, n_val=2000` and a fixed seed for repro.
The resulting directory layout is identical to NB88's subset, so the
existing dataset YAML / loader / collate work unchanged.

Output layout:
    <out_root>/images/{train2017,val2017}/<stem>.jpg
    <out_root>/labels/{train2017,val2017}/<stem>.txt
    <out_root>/lane_targets/{train2017,val2017}/<stem>.pt   (78-D polyline)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _import_p8_prepare(migration_root: Path):
    p = migration_root / 'P8_train' / 'scripts' / 'prepare_bdd_subset.py'
    spec = importlib.util.spec_from_file_location('b_p8_prepare', p)
    if spec is None or spec.loader is None:
        raise ImportError(f'could not load P8 prepare from {p}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('b_p8_prepare', mod)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    here = Path(__file__).resolve()
    migration_root = None
    for cand in here.parents:
        if (cand / 'P0_baseline').exists() and (cand / 'P8_train').exists():
            migration_root = cand
            break
    if migration_root is None:
        raise FileNotFoundError(
            f'Could not locate rmt_ppad_migration/ from {here}'
        )

    p = argparse.ArgumentParser()
    p.add_argument('--out-root', type=Path,
                   default=Path('/content/bdd_subset_10k'))
    p.add_argument('--curve-tar', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/datasets/'
                                'bdd100k_clrkd_curve.tar'))
    p.add_argument('--labels-zip', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/downloads/'
                                'rmt_ppad_weights/BDD_detection_labels.zip'))
    p.add_argument('--lane-tar', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/datasets/'
                                'lane_targets_clr_v1_polyline.tar.gz'))
    p.add_argument('--n-train', type=int, default=10000)
    p.add_argument('--n-val', type=int, default=2000)
    p.add_argument('--seed', type=int, default=89)
    args = p.parse_args()

    prep = _import_p8_prepare(migration_root)
    print(f'[NB89.prep10k] out_root={args.out_root} n_train={args.n_train} n_val={args.n_val}')
    summary = prep.prepare(
        out_root=args.out_root,
        curve_tar=args.curve_tar,
        labels_zip=args.labels_zip,
        lane_tar=args.lane_tar,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )
    (args.out_root / 'prep_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8',
    )
    print('[NB89.prep10k] DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
