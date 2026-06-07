"""Phase B (NB89): brief-training driver for Bezier ablation.

Parallel to `P8_train/scripts/train_lane_only.py`. Differences:
  - Defaults: batch=32, lr0=4e-4 (linear scale from 1e-4 @ batch=8)
  - Accepts ANY model YAML (polyline OR bezier) so the same script runs
    all 3 NB89 rows.
  - 'brief' mode preset: epochs=30, slightly smaller workers/cache
    settings so multiple consecutive launches don't OOM the Colab box.

NB88's debug-record envelope carries over:
  - AMP off (FP16 can't hold the first-epoch DETR loss spikes)
  - warmup_epochs=5, warmup_bias_lr=0
  - LossMonitor with ema_cap_mult on
  - Spatial augmentation off; photometric on
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path hygiene + wandb disable - copied verbatim from train_lane_only.py
# because we MUST do these things before ultralytics is imported.
# ---------------------------------------------------------------------------

def _ensure_vendor_first_on_path(rmt_ppad_root: Path) -> None:
    p = str(rmt_ppad_root.resolve())
    sys.path = [x for x in sys.path if 'ultralytics' not in x.lower()]
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for k in list(sys.modules):
        if k == 'ultralytics' or k.startswith('ultralytics.'):
            del sys.modules[k]


def _disable_wandb_via_settings_file() -> None:
    import hashlib, json as _json, uuid
    cfg_path = Path(os.path.expanduser('~/.config/Ultralytics/settings.json'))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        'settings_version': '0.0.6',
        'datasets_dir': '/content/datasets',
        'weights_dir': '/content/weights',
        'runs_dir': '/content/runs_ultra',
        'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
        'sync': True, 'api_key': '', 'openai_api_key': '',
        'clearml': False, 'comet': False, 'dvc': False, 'hub': False,
        'mlflow': False, 'neptune': False, 'raytune': False,
        'tensorboard': False, 'wandb': False, 'vscode_msg': True,
    }
    cfg_path.write_text(_json.dumps(defaults, indent=2))
    print(f'[train_bezier_brief] disabled wandb via {cfg_path}', flush=True)


def _default_paths():
    here = Path(__file__).resolve()
    # extensions/bezier_lcm/scripts/<file> -> rmt_ppad_migration/
    rmt_root = here.parent.parent.parent.parent / 'vendor' / 'RMT-PPAD'
    return rmt_root


def main() -> int:
    rmt_default = _default_paths()
    p = argparse.ArgumentParser()
    p.add_argument('--rmt-ppad-root', type=Path, default=rmt_default)
    p.add_argument('--model-yaml', type=Path, required=True,
                   help='Path to the model YAML (polyline OR bezier).')
    p.add_argument('--data-yaml', type=Path, required=True,
                   help='Path to the dataset YAML (10k subset).')
    p.add_argument('--name', required=True, help='run name (= results subdir)')
    p.add_argument('--project', default='/content/runs')
    p.add_argument('--device', default='0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-period', type=int, default=10)
    p.add_argument('--val-period', type=int, default=1,
                   help='Validation + save_model cadence. Default 1 overrides '
                        'RMT-PPAD\'s vendored Ultralytics val_period=50 so '
                        'last.pt + metrics fire every epoch.')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr0', type=float, default=4e-4,
                   help='Default 4e-4 = linear scaling of NB88\'s 1e-4 @ batch=8')
    p.add_argument('--lcm-warning-on-row-mismatch', action='store_true',
                   help='Print a warning if the YAML smells like bezier but the '
                        'data root is polyline-targets (and vice versa).')
    p.add_argument('--drive-checkpoint-root', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints'),
                   help='Drive root for per-epoch checkpoint + log mirroring.')
    p.add_argument('--no-drive-sync', action='store_true',
                   help='Disable the per-epoch Drive sync callback.')
    p.add_argument('--lane-match', choices=['hungarian', 'dynamic_k'],
                   default='hungarian',
                   help='Lane prior<->GT matching. hungarian (default) is '
                        '1-to-1 deterministic and fixes the cls collapse.')
    args = p.parse_args()

    # Phase 1: thread the matching scheme via env var (before ultralytics import).
    os.environ['LANE_MATCH'] = args.lane_match
    print(f'[train_bezier_brief] LANE_MATCH={args.lane_match}', flush=True)

    print(f'[train_bezier_brief] name={args.name}  epochs={args.epochs}  '
          f'batch={args.batch}  lr0={args.lr0}')
    print(f'  model = {args.model_yaml}')
    print(f'  data  = {args.data_yaml}')

    if args.lcm_warning_on_row_mismatch:
        ms = str(args.model_yaml).lower()
        ds = str(args.data_yaml).lower()
        if 'bezier' in ms and 'bezier' not in ds:
            print('[train_bezier_brief] WARNING: bezier model YAML paired with '
                  'non-bezier data YAML. Target tensors are likely 78-D and the '
                  'B5 loss will trip.')
        if 'bezier' not in ms and 'bezier' in ds:
            print('[train_bezier_brief] WARNING: polyline model YAML paired '
                  'with bezier data YAML. The P5 loss expects 78-D targets.')

    _disable_wandb_via_settings_file()

    # tqdm throttle + LossMonitor - reuse the P8 module.
    here = Path(__file__).resolve()
    p8_scripts = (here.parent.parent.parent.parent / 'P8_train' / 'scripts').resolve()
    sys.path.insert(0, str(p8_scripts))
    from training_monitors import install_tqdm_throttle, LossMonitor, make_callback
    install_tqdm_throttle(min_iters=50, min_interval=15.0)

    _ensure_vendor_first_on_path(args.rmt_ppad_root)
    from ultralytics import MTDETR

    model = MTDETR(str(args.model_yaml))
    print(f'[train_bezier_brief] model built, '
          f'{sum(p.numel() for p in model.model.parameters()):,} params')

    # Defensive non-leaf buffer sweep (same as P8 train script).
    import torch as _torch
    n_detached = 0
    for name, buf in model.model.named_buffers():
        if buf is None:
            continue
        if buf.requires_grad or not buf.is_leaf:
            with _torch.no_grad():
                clean = buf.detach().clone()
                parts = name.split('.')
                mod = model.model
                for piece in parts[:-1]:
                    mod = getattr(mod, piece)
                setattr(mod, parts[-1], clean)
                n_detached += 1
    if n_detached:
        print(f'[train_bezier_brief] detached {n_detached} non-leaf buffer(s) '
              f'to satisfy ModelEMA.deepcopy')

    out_root = Path(args.project) / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[train_bezier_brief] outputs -> {out_root}')

    # Tee the ENTIRE run to a Drive-mirrored log (see train_lane_only.py).
    try:
        import importlib.util as _ilu0
        _cbp = (Path(__file__).resolve().parent.parent.parent.parent
                / 'training_utils' / 'tools' / 'epoch_callbacks.py')
        _spec0 = _ilu0.spec_from_file_location('epoch_callbacks_tee', _cbp)
        _ec0 = _ilu0.module_from_spec(_spec0); _spec0.loader.exec_module(_ec0)
        _ec0.install_full_log_tee(out_root / 'full_train.log')
    except Exception as _e:  # noqa: BLE001
        print(f'[train_bezier_brief] WARNING: full-log tee not installed: {_e}', flush=True)

    monitor = LossMonitor(
        ema_decay=0.99,
        spike_threshold=5.0,
        spike_count=50,
        warmup_batches=500,        # < NB88's 1000 since the brief budget
                                   # is only ~310 batches per epoch
        divergence_window=500,
        divergence_growth=3.0,
        log_every=100,
        ema_cap_mult=10.0,         # iter-1 fix from debug_record.md
    )
    cb_name, cb = make_callback(monitor, also_print_every=500)
    model.add_callback(cb_name, cb)

    # Per-epoch Drive sync + metrics-table callbacks (shared module with
    # train_lane_only.py so the on-screen format is identical across runs).
    try:
        import importlib.util as _ilu
        # extensions/bezier_lcm/scripts/<file> -> rmt_ppad_migration/
        cb_path = (Path(__file__).resolve().parent.parent.parent.parent
                   / 'training_utils' / 'tools' / 'epoch_callbacks.py')
        spec = _ilu.spec_from_file_location('epoch_callbacks_mod', cb_path)
        eclbk = _ilu.module_from_spec(spec); spec.loader.exec_module(eclbk)
        tbl_name, tbl_cb = eclbk.make_metrics_table_callback()
        model.add_callback(tbl_name, tbl_cb)
        print(f'[train_bezier_brief] metrics_table registered as {tbl_name}', flush=True)
        if not args.no_drive_sync:
            ds_name, ds_cb = eclbk.make_drive_sync_callback(
                drive_checkpoint_root=args.drive_checkpoint_root,
                run_name=args.name,
            )
            model.add_callback(ds_name, ds_cb)
            print(f'[train_bezier_brief] drive_sync -> {args.drive_checkpoint_root}/{args.name}', flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'[train_bezier_brief] WARNING: failed to register epoch callbacks: {e}', flush=True)

    results = model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save_period=args.save_period,
        # CRITICAL: RMT-PPAD's vendored Ultralytics defaults val_period=50.
        # That gates BOTH validate() AND save_model() in trainer.py L529.
        # With our 30-epoch brief budget val never runs, trainer.metrics
        # stays empty, last.pt is never written, and our per-epoch
        # callbacks (drive_sync, metrics_table) have nothing to read.
        # Default 1 here so validation + save fire every epoch; pass
        # --val-period N on the CLI to ablate.
        val_period=args.val_period,
        lr0=args.lr0,
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        amp=False,
        warmup_epochs=5.0,
        warmup_bias_lr=0.0,
        warmup_momentum=0.8,
        verbose=True,
        # deterministic=False silences the nondeterministic-algorithm
        # UserWarnings (nll_loss2d / grid_sampler / histc) that Ultralytics'
        # default deterministic=True triggers; see train_lane_only.py.
        deterministic=False,
        # Spatial transforms OFF (would desync seg mask + lane targets).
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    )

    summary = {
        'name': args.name,
        'model_yaml': str(args.model_yaml),
        'data_yaml': str(args.data_yaml),
        'epochs': args.epochs, 'batch': args.batch, 'lr0': args.lr0,
        'workers': args.workers, 'imgsz': args.imgsz,
        'project': args.project,
    }
    (out_root / 'train_args.json').write_text(json.dumps(summary, indent=2))
    print('\n[train_bezier_brief] DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
