"""Re-validate a saved checkpoint with the FIXED curveIoU metric (val-only).

NB98 reported curveIoU=0 for every row because of two metric bugs (now fixed
in P7_validator/tools/lane_curve_f1.py): (1) curveIoU was scored only over the
top-8-by-score priors, which a near-flat cls makes a fixed clustered set that
misses every GT; (2) predicted lanes were dropped by `row[1] < 0.5`, treating
the cls LOGIT as a 0/1 validity flag. Both are METRIC changes, so the corrected
ranking needs only a VALIDATION pass over each saved best.pt -- NOT a retrain.
The `[lane-f1] ... mean-best-IoU=` line printed by val.get_stats is the
corrected curveIoU.

Usage:
    python revalidate_curveiou.py --weights <best.pt> \
        --model-yaml <rtdetr-l_bdd_clr_lane.yaml> --data-yaml <BDD_lane_only_10k.yaml>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_vendor_first_on_path(rmt_ppad_root: Path) -> None:
    p = str(rmt_ppad_root.resolve())
    sys.path = [x for x in sys.path if 'ultralytics' not in x.lower()]
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for k in list(sys.modules):
        if k == 'ultralytics' or k.startswith('ultralytics.'):
            del sys.modules[k]


def main() -> int:
    here = Path(__file__).resolve()
    mig = here.parent.parent.parent  # P8_train/scripts/ -> rmt_ppad_migration/
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=Path, required=True, help='best.pt / last.pt')
    ap.add_argument('--model-yaml', type=Path,
                    default=mig / 'vendor/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd_clr_lane.yaml')
    ap.add_argument('--data-yaml', type=Path,
                    default=mig / 'vendor/RMT-PPAD/ultralytics/cfg/datasets/BDD_lane_only_10k.yaml')
    ap.add_argument('--rmt-ppad-root', type=Path, default=mig / 'vendor' / 'RMT-PPAD')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default='0')
    ap.add_argument('--name', default='reval')
    ap.add_argument('--use-aux-seg', action='store_true',
                    help='Required for the abl_aux_seg checkpoint: the decoder '
                         'reads USE_AUX_SEG at build time, so the aux submodule '
                         'must be enabled to load those weights.')
    args = ap.parse_args()

    if not args.weights.exists():
        print(f'[reval] MISSING weights: {args.weights}\n'
              f'        (best.pt should be on Drive at training_runs/checkpoints/<row>/; '
              f'if absent, that row must be retrained.)', flush=True)
        return 2

    # MUST be set before MTDETR import/build (decoder reads it in __init__).
    if args.use_aux_seg:
        os.environ['USE_AUX_SEG'] = '1'
        os.environ.setdefault('AUX_SEG_CLASSES', '2')
    # Same env knobs the rows trained under, so the lane loss/val config matches.
    os.environ.setdefault('LANE_MATCH', 'hungarian')
    os.environ.setdefault('LANE_DIFF_CLAMP', '100')

    _ensure_vendor_first_on_path(args.rmt_ppad_root)
    from ultralytics import MTDETR

    print(f'[reval] loading {args.weights}', flush=True)
    model = MTDETR(str(args.weights))

    # Standalone val (model.val) routes through AutoBackend, which calls
    # model.fuse(). MTDETR's inherited fuse() trips with
    # "'DetectionModel' object has no attribute 'model'", and fusion is ONLY an
    # inference-speed optimization -- it has no effect on val metrics, and
    # training-time validation never fuses either. Disable it so the standalone
    # path matches the proven training path.
    try:
        _inner = model.model
        _inner.fuse = (lambda *a, **k: _inner)
        _inner.is_fused = (lambda *a, **k: True)
        print('[reval] fuse() disabled for standalone val (no effect on metrics)', flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f'[reval] WARNING: could not disable fuse(): {_e}', flush=True)

    print(f'[reval] val on {args.data_yaml.name} (watch [lane-f1] ... mean-best-IoU=)', flush=True)
    try:
        metrics = model.val(
            data=str(args.data_yaml), imgsz=args.imgsz, batch=args.batch,
            device=args.device, name=args.name, verbose=True,
        )
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f'\n[reval] standalone val FAILED ({type(e).__name__}: {e}).\n'
              f'        FALLBACK: re-run this row\'s NB98 training cell instead -- its\n'
              f'        per-epoch validation uses the proven trainer path and now\n'
              f'        computes the fixed curveIoU live (the metric fix is in the\n'
              f'        imported module either way).', flush=True)
        return 3
    # model.val()'s results_dict carries only DETECTION keys, so the lane
    # metrics read None here. The authoritative lane numbers are in the
    # validator's `[lane-f1] ... mean-best-IoU=` LOG line printed just above
    # (the notebook parses that). Surface whatever the dict does expose, and
    # point at the log line so the value is never reported as a bogus None.
    rd = getattr(metrics, 'results_dict', None) or {}
    cur = rd.get('metrics/lane_curveIoU(lane)')
    print(f'\n[reval] DONE {args.name}: '
          f"curveIoU={cur if cur is not None else 'see [lane-f1] mean-best-IoU above'} "
          f"(results_dict lane keys may be absent; the [lane-f1] log line is "
          f"authoritative)", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
