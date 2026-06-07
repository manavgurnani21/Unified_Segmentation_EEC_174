"""Phase P8: training driver for the lane-only MTDETR variant.

Two modes:
  --mode smoke  (epochs=2, batch=4): pipeline integration test
  --mode full   (epochs=250, batch=8): the real ablation row

The script:
  1. Ensures the vendored RMT-PPAD ultralytics is on sys.path (NOT a pip
     install of plain ultralytics).
  2. Builds an MTDETR model from the model YAML (defaults to the lane-only
     rtdetr-l_bdd_clr_lane.yaml from P4).
  3. Calls `.train(...)` with the data YAML (defaults to BDD_lane_only.yaml
     from P6).
  4. Writes per-epoch loss history to results/<run_name>/losses.json so
     the ablation aggregator can build the final table.

Hyperparameters honor the CLRKDNet defaults from configs/DLA_CULane.py:
  optimizer: AdamW, lr=6e-4 (smoke uses 1e-4 to converge faster)
  workers: 4
"""
from __future__ import annotations

import argparse
import json
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


def _disable_wandb_via_settings_file() -> None:
    """Pre-write Ultralytics' settings.json with wandb=False BEFORE the
    first ultralytics import.

    Why this matters: `ultralytics/utils/callbacks/wb.py` only registers
    wandb callbacks if `SETTINGS["wandb"] is True` at MODULE IMPORT TIME
    (the `callbacks = (...) if wb else {}` ternary at the file's bottom).
    Once those callbacks are registered, `MTDETR.train(project=...)` tries
    to use the project as a wandb project NAME and wandb's settings
    validator rejects anything containing a slash - the path we pass to
    Ultralytics for the OUTPUT DIRECTORY ('/content/runs/train') is
    invalid as a wandb project name.

    Skipping the callback entirely is cleaner than monkey-patching wandb's
    validator. Ultralytics validates the settings file at import; if all
    keys / types / version match its defaults, our wandb=False override
    survives.
    """
    import hashlib
    import json
    import uuid

    cfg_path = Path(os.path.expanduser('~/.config/Ultralytics/settings.json'))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        # Mirror ultralytics/utils/__init__.py SettingsManager defaults at
        # v0.0.6. If the version drifts the validator will reset and our
        # override will be lost; that's the user's signal to re-pin.
        'settings_version': '0.0.6',
        'datasets_dir': '/content/datasets',
        'weights_dir': '/content/weights',
        'runs_dir': '/content/runs_ultra',
        'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
        'sync': True,
        'api_key': '',
        'openai_api_key': '',
        # All third-party loggers off so callbacks don't accidentally fire.
        'clearml': False,
        'comet': False,
        'dvc': False,
        'hub': False,
        'mlflow': False,
        'neptune': False,
        'raytune': False,
        'tensorboard': False,
        'wandb': False,            # <-- the one that fixes the error
        'vscode_msg': True,
    }
    cfg_path.write_text(json.dumps(defaults, indent=2))
    print(f'[train_lane_only] disabled wandb via {cfg_path}', flush=True)


def _default_paths():
    here = Path(__file__).resolve()
    rmt_ppad_root = here.parent.parent.parent / 'vendor' / 'RMT-PPAD'
    model_yaml = (
        rmt_ppad_root / 'ultralytics' / 'cfg' / 'models' / 'mt-detr'
        / 'rtdetr-l_bdd_clr_lane.yaml'
    )
    data_yaml = (
        rmt_ppad_root / 'ultralytics' / 'cfg' / 'datasets' / 'BDD_lane_only.yaml'
    )
    return rmt_ppad_root, model_yaml, data_yaml


def main() -> int:
    default_rmt, default_model, default_data = _default_paths()
    p = argparse.ArgumentParser()
    p.add_argument('--rmt-ppad-root', type=Path, default=default_rmt)
    p.add_argument('--model-yaml', type=Path, default=default_model)
    p.add_argument('--data-yaml', type=Path, default=default_data)
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--name', default='clr_lane_default')
    p.add_argument('--project', default='/content/runs/train')
    p.add_argument('--device', default='0',
                   help='"cpu" or comma-separated GPU ids')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save-period', type=int, default=10)
    p.add_argument('--fliplr', type=float, default=0.5,
                   help='Horizontal-flip probability. SAFE now: RandomFlip '
                        'flips the CLR lane_targets in sync. The original '
                        'RMT-PPAD trains with flip ON; we had it off, the main '
                        'overfitting cause. Set 0.0 to disable.')
    p.add_argument('--weight-decay', type=float, default=0.05,
                   help='AdamW weight decay (Ultralytics default 0.0005). '
                        'Raised to 0.05 as an extra regularizer against the '
                        'detection overfitting.')
    p.add_argument('--freeze-trunk-after', type=int, default=-1,
                   help='If >=0, freeze the shared backbone+neck at the start '
                        'of this epoch (two-phase fix for lane negative '
                        'transfer; NB97 lanes peaked ~ep11 then the trunk '
                        'drifted to detection). -1 (default) = never freeze. '
                        'Try ~10-12 (just after the lane peak).')
    p.add_argument('--patience', type=int, default=30,
                   help='Early-stopping patience (epochs without fitness '
                        'improvement). NB88/89 peaked at ep~17-20 then '
                        'declined; patience=30 stops the run ~30 epochs after '
                        'the peak so we do not burn 200 wasted epochs, while '
                        'still giving the slower lane branch room. best.pt '
                        '(the peak) is preserved regardless.')
    p.add_argument('--resume', default='',
                   help="Resume an interrupted run instead of starting fresh. "
                        "Value is a checkpoint path, or 'auto' to use "
                        '{project}/{name}/weights/last.pt. Ultralytics restores '
                        'the optimizer + epoch + cos-LR + EarlyStopping + '
                        'best_fitness state from the .pt and continues to '
                        '--epochs (e.g. NB107 laneiou_both_full stopped at ep18 '
                        '-> resume to 120). IMPORTANT on Colab: /content/ is '
                        'wiped between sessions, so pass the DRIVE path '
                        'explicitly (e.g. --resume /content/drive/MyDrive/.../'
                        'laneiou_both_full/last.pt); the {project}/{name} '
                        'last.pt will not exist in a fresh session. The '
                        "checkpoint's OWN saved hyperparameters are reloaded "
                        '(trainer.check_resume replaces args), so changing CLI '
                        'lr0/patience here has no effect on resume; only '
                        'imgsz/batch/device/epochs override. Lane levers are '
                        'threaded via env vars at loss-construct time, so '
                        '--lane-y-reweight DOES take effect on resume.')
    p.add_argument('--val-period', type=int, default=1,
                   help='How often to run validation + save model (every N epochs). '
                        'RMT-PPAD ships with val_period=50 by default; we override '
                        'to 1 so last.pt/best.pt and the metrics table fire every '
                        'epoch. Increase (e.g. 5) for long runs if val cost matters.')
    p.add_argument('--epochs', type=int, default=None,
                   help='override the default (2 for smoke, 250 for full)')
    p.add_argument('--batch', type=int, default=None,
                   help='override the default (4 for smoke, 8 for full)')
    p.add_argument('--lr0', type=float, default=None,
                   help='override the default (1e-4 for smoke, 6e-4 for full)')
    p.add_argument('--drive-checkpoint-root', type=Path,
                   default=Path('/content/drive/MyDrive/EcoCAR/training_runs/checkpoints'),
                   help='Drive root for per-epoch checkpoint + log mirroring. '
                        'A subdir <run_name>/ will be created automatically.')
    p.add_argument('--no-drive-sync', action='store_true',
                   help='Disable the per-epoch Drive sync callback.')
    p.add_argument('--lane-match', choices=['hungarian', 'dynamic_k'],
                   default='hungarian',
                   help='Lane prior<->GT matching. hungarian (default) is '
                        '1-to-1 deterministic and fixes the cls collapse; '
                        'dynamic_k is the original SimOTA scheme (frozen IoU).')
    # NB95/NB96 diagnostic knobs (forwarded to MTDETRDLoss via env vars).
    p.add_argument('--lane-weights', choices=['cut5x', 'clrkd'],
                   default='cut5x',
                   help='cut5x = joint-stable 5x-reduced lane weights (default); '
                        'clrkd = CLRKDNet full weights (stronger geometry grad).')
    p.add_argument('--diff-clamp', default='100',
                   help="xytl smooth-l1 diff clamp in px, or 'none' to disable "
                        "(the clamp may be zeroing the geometry gradient).")
    # ---- Sprint-1 lane-head levers (NEXT_STEPS sec 5), forwarded to
    # MTDETRDLoss via env. All default OFF => NB101-identical baseline. ----
    p.add_argument('--lane-iou-cls', default='off',
                   help="S1.3 IoU-aware cls: 'off' (hard 1 target) or a QFL "
                        "gamma like '2.0' (soft target = line-IoU, spreads "
                        "score_std so the S0.3 threshold is stable).")
    p.add_argument('--lane-smooth-w', type=float, default=0.0,
                   help='S1.2 curvature (2nd-difference) smoothness weight on '
                        'matched lane x-offsets. 0 = off (default; S0.4 found '
                        'near already smoother than far, so this is a re-check).')
    p.add_argument('--lane-y-reweight', choices=['none', 'near', 'angle'],
                   default='none',
                   help="S1.4 y-reweight of the per-row x error. 'near' weights "
                        'near-camera (bottom) rows heavier to stop the '
                        'length-shrink; angle normalizes by local dx/dy.')
    p.add_argument('--lane-len-hinge', type=float, default=0.0,
                   help='S1.4b weight on an ASYMMETRIC too-short hinge on the '
                        'predicted length field: penalizes (gt_len - pred_len)_+ '
                        'only (normalized to [0,1] by n_strips). The xytl '
                        'smooth-L1 supervises length symmetrically and weakly, so '
                        'the model shrinks lanes (NB107 length_mean 0.155->0.094); '
                        'this DIRECTLY raises the cost of too-short lanes. 0 = off '
                        '(default). Try ~1-3; pairs with --lane-y-reweight near.')
    p.add_argument('--lane-eval-tau', type=float, default=0.0,
                   help='S1.1 eval confidence threshold for the curve-F1 metric '
                        '(softmax_pos > tau before top-max_lanes). 0 = old top-8. '
                        'Sprint-0 S0.3 found ~0.4 best on the current model.')
    p.add_argument('--lane-iou-type', choices=['line', 'laneiou'], default='line',
                   help="S2.A: 'line' = fixed-band CLRNet LineIoU (default); "
                        "'laneiou' = CLRerNet angle-aware LaneIoU in the "
                        'regression LOSS (widens the IoU band on steep/near-'
                        'field rows).')
    p.add_argument('--lane-iou-match', choices=['follow', 'line', 'laneiou'],
                   default='follow',
                   help="S2.A: LaneIoU in the MATCHER cost, gated separately "
                        "from the loss. 'follow' (default) = same as "
                        "--lane-iou-type; set 'line'/'laneiou' to isolate "
                        'loss-only vs matcher-only vs both in the ablation.')
    p.add_argument('--lane-iou-cost-w', type=float, default=2.0,
                   help='S2.A: weight of the LaneIoU reward in the hungarian '
                        'matcher cost (only when --lane-iou-type laneiou).')
    # ---- MTL anti-negative-transfer ablation knobs (training_utils/tools/
    # mtl_techniques.py). All default OFF so the baseline is unchanged. ----
    p.add_argument('--pcgrad', action='store_true',
                   help='MTL#1 Gradient surgery (PCGrad): project the detection '
                        'backbone gradient off the lane gradient when they '
                        'conflict, so detection cannot unlearn lane features. '
                        'Requires amp=False (the standing config). ~2x backward cost.')
    p.add_argument('--backbone-lr-mult', type=float, default=1.0,
                   help='MTL#2 Asymmetric LR: scale the shared backbone+neck '
                        'learning rate by this factor (e.g. 0.1) while heads stay '
                        'at base lr. Stops detection thrashing the shared FPN. '
                        '1.0 = off.')
    p.add_argument('--backbone-boundary', type=int, default=27,
                   help='Layer index (inclusive) that separates the shared '
                        'backbone+neck (<=) from the task heads (>). Used by '
                        '--backbone-lr-mult and --pcgrad. Matches the trainer '
                        "get_backbone_params default (27).")
    p.add_argument('--seg-gate-floor', type=float, default=-1.0,
                   help='MTL#3 Strengthen GCA: raise the lane LiteDynamicGate '
                        'clamp_min to this floor (e.g. 0.3) so the lane branch '
                        'injects more task-specific transform instead of staying '
                        'coupled to the detection-dominated trunk. <0 = off '
                        '(keep the built-in 0.05).')
    p.add_argument('--det-decay-epoch', type=int, default=-1,
                   help='MTL#4 Dynamic loss weighting: at this epoch, multiply '
                        'the detection loss by --det-decay-factor (hand gradient '
                        'priority to the lane head). <0 = off.')
    p.add_argument('--det-decay-factor', type=float, default=0.5,
                   help='Factor applied to L_det at --det-decay-epoch (default 0.5).')
    # MTL#6 Auxiliary dense segmentation (training-only). Re-attaches the
    # native RMT-PPAD drivable+lane dense decoder to the shared FPN features
    # to inject dense gradients that fight detection domination. Bypassed at
    # inference (zero overhead). drivable term only fires when drivable GT is
    # present (subset must be rebuilt with drivable masks).
    p.add_argument('--use-aux-seg', action='store_true',
                   help='MTL#6: enable training-only auxiliary dense drivable+lane '
                        'segmentation heads (negative-transfer regularizer).')
    p.add_argument('--aux-seg-classes', type=int, default=2,
                   help='Aux dense seg output channels (2 = drivable+lane; 1 = lane only).')
    p.add_argument('--aux-drivable-weight', type=float, default=0.5,
                   help='Weight on the aux drivable-area dense loss.')
    p.add_argument('--aux-lane-weight', type=float, default=0.5,
                   help='Weight on the aux lane dense loss.')
    args = p.parse_args()

    # Phase 1 / diagnostics: thread knobs to MTDETRDLoss via env vars (set
    # BEFORE any ultralytics import so the loss module reads them at init).
    os.environ['LANE_MATCH'] = args.lane_match
    os.environ['LANE_WEIGHT_PRESET'] = args.lane_weights
    os.environ['LANE_DIFF_CLAMP'] = str(args.diff_clamp)
    # Sprint-1 levers (env-driven, read by MTDETRDLoss.__init__).
    os.environ['LANE_IOU_CLS'] = str(args.lane_iou_cls)
    os.environ['LANE_SMOOTH_W'] = str(args.lane_smooth_w)
    os.environ['LANE_Y_REWEIGHT'] = str(args.lane_y_reweight)
    os.environ['LANE_LEN_HINGE_W'] = str(args.lane_len_hinge)   # S1.4b too-short hinge
    os.environ['LANE_EVAL_TAU'] = str(args.lane_eval_tau)
    os.environ['LANE_IOU_TYPE'] = str(args.lane_iou_type)        # S2.A (loss)
    if args.lane_iou_match != 'follow':                          # S2.A (matcher)
        os.environ['LANE_IOU_MATCH'] = str(args.lane_iou_match)
    else:
        os.environ.pop('LANE_IOU_MATCH', None)  # follow LANE_IOU_TYPE
    os.environ['LANE_IOU_COST_W'] = str(args.lane_iou_cost_w)
    print(f'[train_lane_only] LANE_MATCH={args.lane_match} '
          f'LANE_WEIGHT_PRESET={args.lane_weights} '
          f'LANE_DIFF_CLAMP={args.diff_clamp} '
          f'| S1: iou_cls={args.lane_iou_cls} smooth_w={args.lane_smooth_w} '
          f'y_reweight={args.lane_y_reweight} len_hinge={args.lane_len_hinge}',
          flush=True)

    # MTL#6 aux dense seg: thread on/off + weights via env BEFORE the
    # ultralytics import, so MTDETRDecoder.__init__ constructs the aux head
    # and MTDETRDLoss.__init__ reads the weights. OFF by default = unchanged.
    if args.use_aux_seg:
        os.environ['USE_AUX_SEG'] = '1'
        os.environ['AUX_SEG_CLASSES'] = str(args.aux_seg_classes)
        os.environ['AUX_DRIVABLE_WEIGHT'] = str(args.aux_drivable_weight)
        os.environ['AUX_LANE_WEIGHT'] = str(args.aux_lane_weight)
        print(f'[train_lane_only] USE_AUX_SEG=1 classes={args.aux_seg_classes} '
              f'drivable_w={args.aux_drivable_weight} lane_w={args.aux_lane_weight} '
              f'(training-only dense aux)', flush=True)
    else:
        os.environ.pop('USE_AUX_SEG', None)

    if args.mode == 'smoke':
        epochs = args.epochs if args.epochs is not None else 2
        batch = args.batch if args.batch is not None else 4
        lr0 = args.lr0 if args.lr0 is not None else 1e-4
    else:
        # Epoch budget cut 250 -> 120. The NB88/89 logs showed detection
        # mAP peaks at epoch ~17-20 then DECLINES for 200+ epochs (spatial
        # aug is off for mask consistency, so the det branch is under-
        # regularized and overfits fast). 250 epochs was wasteful AND the
        # final weights were worse than the peak. 120 + early stopping
        # (--patience) keeps the run near the peak while still giving the
        # slower lane branch room to converge. best.pt (the peak) is the
        # deliverable - it is preserved by Ultralytics + the drive_sync.
        epochs = args.epochs if args.epochs is not None else 120
        batch = args.batch if args.batch is not None else 8
        # P8: cut from CLRKDNet's 6e-4 -> 1e-4. The 35M-param RT-DETR +
        # CLR lane head combo trains the encoder/decoder from scratch and
        # was diverging in epoch 1 with the higher rate. 1e-4 is in the
        # range RT-DETR's own train.py uses for AdamW.
        lr0 = args.lr0 if args.lr0 is not None else 1e-4

    print(f'[train_lane_only] mode={args.mode}  epochs={epochs}  batch={batch}  lr0={lr0}')
    print(f'  model  = {args.model_yaml}')
    print(f'  data   = {args.data_yaml}')
    print(f'  device = {args.device}')

    # P8: must disable wandb BEFORE importing ultralytics so the callback
    # module's import-time `if wb` check skips registration. Otherwise
    # MTDETR.train(project='/content/runs/train') trips wandb's project-
    # name validator (slashes forbidden).
    _disable_wandb_via_settings_file()

    # P8: throttle ALL tqdm progress bars so we see ~88 updates/epoch
    # instead of ~8750. Must run BEFORE ultralytics imports tqdm.
    sys.path.insert(0, str(Path(__file__).parent))
    from training_monitors import install_tqdm_throttle, LossMonitor, make_callback
    install_tqdm_throttle(min_iters=100, min_interval=15.0)

    _ensure_vendor_first_on_path(args.rmt_ppad_root)
    from ultralytics import MTDETR  # noqa: E402

    model = MTDETR(str(args.model_yaml))
    print(f'[train_lane_only] model built, '
          f'{sum(p.numel() for p in model.model.parameters()):,} params')

    # Belt-and-suspenders for the ModelEMA deepcopy issue (P8 fix). Any
    # buffer that is non-leaf with requires_grad=True will break the
    # one-time deepcopy that builds the EMA. CLRHead's priors are the
    # known offender (the LaneSegHead constructor detaches them), but
    # detach any others we missed.
    import torch as _torch
    n_detached = 0
    for name, buf in model.model.named_buffers():
        if buf is None:
            continue
        if buf.requires_grad or not buf.is_leaf:
            with _torch.no_grad():
                clean = buf.detach().clone()
                # Walk to the module that owns it and reassign the buffer.
                parts = name.split('.')
                mod = model.model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], clean)
                n_detached += 1
                print(f'  [defensive] detached non-leaf buffer: {name}')
    if n_detached:
        print(f'[train_lane_only] detached {n_detached} non-leaf buffer(s) to satisfy ModelEMA.deepcopy')

    out_root = Path(args.project) / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[train_lane_only] outputs -> {out_root}')

    # Tee the ENTIRE run (model summary, '[MTDETRDLoss] lane matching
    # scheme = ...', every warning, every epoch row) to a file the
    # drive_sync callback mirrors to Drive each epoch. Install it now,
    # before model.train(), so all of training is captured even if the
    # run_streaming notebook log fails to flush or Colab disconnects.
    try:
        import importlib.util as _ilu0
        _cbp = (Path(__file__).resolve().parent.parent.parent
                / 'training_utils' / 'tools' / 'epoch_callbacks.py')
        _spec0 = _ilu0.spec_from_file_location('epoch_callbacks_tee', _cbp)
        _ec0 = _ilu0.module_from_spec(_spec0); _spec0.loader.exec_module(_ec0)
        _ec0.install_full_log_tee(out_root / 'full_train.log')
    except Exception as _e:  # noqa: BLE001
        print(f'[train_lane_only] WARNING: full-log tee not installed: {_e}', flush=True)

    # P8: hook the loss-anomaly monitor onto the trainer. Stops training
    # automatically on NaN/Inf or sustained spike/divergence.
    #
    # Tolerance tuning (after the first divergent run on 5x-larger weights):
    #   warmup_batches: 200 -> 1000 — the lane branch has 192 priors that
    #     have to settle via SimOTA assignment + LineIoU regression from
    #     scratch. The first ~1 epoch (~8750 batches) is a legitimate ramp,
    #     not divergence. 1000 batches lets the EMA stabilize before we
    #     start watching its trend.
    #   divergence_growth: 1.5 -> 3.0 — with the now-5x-smaller lane
    #     weights, ll_seg starts ~1-3 and can climb to ~10 during settling.
    #     1.5x over a 500-batch window was tripping on healthy learning;
    #     3.0x still catches a real blow-up but tolerates the ramp.
    monitor = LossMonitor(
        ema_decay=0.99,
        spike_threshold=5.0,     # 5x EMA tolerated
        spike_count=50,          # ... but for at most 50 consecutive batches
        warmup_batches=1000,     # skip divergence check while losses settle
        divergence_window=500,   # then watch the EMA trend over 500-step windows
        divergence_growth=3.0,   # EMA growing 3x over the window = trouble
        log_every=100,
    )
    cb_name, cb = make_callback(monitor, also_print_every=500)
    model.add_callback(cb_name, cb)
    print(f'[train_lane_only] LossMonitor registered as {cb_name} callback', flush=True)

    # Two-phase TRUNK FREEZE (fixes lane negative transfer). NB97 showed
    # lane IoU/subAcc peak at ep~11 then decline while detection mAP keeps
    # rising and Det_gate climbs to 0.65 but Seg_gate stays ~0.54 - i.e.
    # the lane branch stays coupled to the shared backbone, which keeps
    # being optimized for detection and overwrites lane features. If
    # --freeze-trunk-after N >= 0, this callback freezes the shared
    # backbone+neck (model[0:28]) at the START of epoch N, so the trunk is
    # locked at its lane-friendly state and ONLY the task decoders + GCA
    # adapters + lane head keep training. Detection won't degrade (its
    # decoder still trains); lanes can keep climbing on a stable trunk.
    if args.freeze_trunk_after is not None and args.freeze_trunk_after >= 0:
        _freeze_state = {'done': False}

        def _freeze_trunk_cb(trainer, _st=_freeze_state, _after=args.freeze_trunk_after):
            ep = getattr(trainer, 'epoch', 0)
            if _st['done'] or ep < _after:
                return
            seq = getattr(trainer.model, 'model', None)
            if seq is None:
                return
            n_layers = len(seq)
            n_frozen_layers = max(0, n_layers - 1)  # all but the final MTDETRDecoder
            n_tensors = 0
            for i in range(n_frozen_layers):
                seq[i].eval()  # also stop BN running-stat drift
                for p in seq[i].parameters():
                    if p.requires_grad:
                        p.requires_grad_(False)
                        n_tensors += 1
            _st['done'] = True
            print(f'\n[freeze-trunk] froze shared backbone+neck layers 0..{n_frozen_layers-1} '
                  f'at epoch {ep + 1} ({n_tensors} tensors). Detection decoder + GCA + lane '
                  f'head keep training on the locked trunk.', flush=True)

        model.add_callback('on_train_epoch_start', _freeze_trunk_cb)
        print(f'[train_lane_only] trunk-freeze armed for epoch {args.freeze_trunk_after}', flush=True)

    # ---- MTL anti-negative-transfer knobs (techniques 1-4; #5 freeze above).
    # All default OFF, so omitting the flags reproduces the baseline exactly.
    try:
        import importlib.util as _ilu_mtl
        _mtl_path = (Path(__file__).resolve().parent.parent.parent
                     / 'training_utils' / 'tools' / 'mtl_techniques.py')
        _spec_mtl = _ilu_mtl.spec_from_file_location('mtl_techniques', _mtl_path)
        mtl = _ilu_mtl.module_from_spec(_spec_mtl); _spec_mtl.loader.exec_module(mtl)
    except Exception as _e:  # noqa: BLE001
        mtl = None
        print(f'[train_lane_only] WARNING: mtl_techniques not loaded: {_e}', flush=True)

    if mtl is not None:
        mtl.reset_det_loss_scale()  # clear stale DET_LOSS_SCALE from a reused process

        # #1 PCGrad — the trainer reads MTL_PCGRAD each step.
        if args.pcgrad:
            os.environ['MTL_PCGRAD'] = '1'
            os.environ['MTL_PCGRAD_LAYERS'] = str(args.backbone_boundary)
            print(f'[train_lane_only] MTL#1 PCGrad ON (boundary={args.backbone_boundary}; '
                  f'needs amp=False) - ~2x backward cost', flush=True)
        else:
            os.environ.pop('MTL_PCGRAD', None)

        # #2 Asymmetric LR — monkeypatch the trainer's optimizer builder
        # (must happen before model.train() builds the optimizer).
        if args.backbone_lr_mult != 1.0:
            try:
                from ultralytics.models.mtdetr.train import MTDETRTrainer  # noqa: E402
                mtl.install_asymmetric_lr(MTDETRTrainer, args.backbone_boundary,
                                          args.backbone_lr_mult)
            except Exception as _e:  # noqa: BLE001
                print(f'[train_lane_only] WARNING: asym-LR install failed: {_e}', flush=True)

        # #3 Strengthen GCA — raise the lane gate clamp floor.
        if args.seg_gate_floor is not None and args.seg_gate_floor >= 0.0:
            mtl.set_seg_gate_floor(model.model, args.seg_gate_floor)

        # #4 Dynamic det-loss decay — callback flips DET_LOSS_SCALE at epoch N.
        if args.det_decay_epoch is not None and args.det_decay_epoch >= 0:
            dd_name, dd_cb = mtl.make_det_decay_callback(args.det_decay_epoch,
                                                         args.det_decay_factor)
            model.add_callback(dd_name, dd_cb)
            print(f'[train_lane_only] MTL#4 det-decay armed: L_det x{args.det_decay_factor} '
                  f'at epoch {args.det_decay_epoch}', flush=True)

    # Per-epoch Drive sync + metrics-table callbacks. Both fire after
    # the validator populates trainer.metrics, so the table sees the
    # latest val numbers and the sync writes the just-saved best.pt.
    try:
        import importlib.util as _ilu
        cb_path = (Path(__file__).resolve().parent.parent.parent
                   / 'training_utils' / 'tools' / 'epoch_callbacks.py')
        spec = _ilu.spec_from_file_location('epoch_callbacks_mod', cb_path)
        eclbk = _ilu.module_from_spec(spec); spec.loader.exec_module(eclbk)
        tbl_name, tbl_cb = eclbk.make_metrics_table_callback()
        model.add_callback(tbl_name, tbl_cb)
        print(f'[train_lane_only] metrics_table registered as {tbl_name}', flush=True)
        if not args.no_drive_sync:
            ds_name, ds_cb = eclbk.make_drive_sync_callback(
                drive_checkpoint_root=args.drive_checkpoint_root,
                run_name=args.name,
            )
            model.add_callback(ds_name, ds_cb)
            print(f'[train_lane_only] drive_sync -> {args.drive_checkpoint_root}/{args.name}', flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'[train_lane_only] WARNING: failed to register epoch callbacks: {e}', flush=True)

    # Resume resolution. '' = fresh run (False). 'auto' = the standard
    # {project}/{name}/weights/last.pt. Anything else = explicit checkpoint
    # path (on Colab, pass the DRIVE path because /content/ is wiped between
    # sessions). Ultralytics' trainer.check_resume() restores optimizer +
    # epoch + best_fitness + EMA from the .pt and replaces self.args with the
    # checkpoint's saved hyperparameters (only imgsz/batch/device/epochs from
    # this call still override). Lane levers bypass that path (env vars read
    # at loss-construct time), so --lane-y-reweight etc. DO apply on resume.
    _resume_arg = False
    if str(args.resume).strip():
        if str(args.resume).strip().lower() == 'auto':
            _resume_arg = str(Path(args.project) / args.name / 'weights' / 'last.pt')
        else:
            _resume_arg = str(args.resume).strip()
        if not Path(_resume_arg).exists():
            raise FileNotFoundError(
                f'[train_lane_only] --resume checkpoint not found: {_resume_arg}\n'
                '  On Colab /content/ is wiped between sessions; pass the DRIVE '
                'path to last.pt explicitly, e.g.\n'
                '  --resume /content/drive/MyDrive/<...>/laneiou_both_full/last.pt'
            )
        print(f'[train_lane_only] RESUME from {_resume_arg} '
              f'(optimizer/epoch/best_fitness restored; epochs target={epochs})',
              flush=True)

    # P8 fix: in lane-only mode, the rasterized `lane_seg_mask` is built
    # from `lane_targets` and not transformed alongside the image. So we
    # MUST disable spatial augmentations that would invalidate the
    # img <-> seg_mask + img <-> lane_targets consistency. Photometric
    # ones (HSV, blur, etc.) only touch pixel values and stay safe.
    results = model.train(
        data=str(args.data_yaml),
        epochs=epochs,
        # Resume from last.pt when --resume is set (False = fresh run).
        # check_resume() reads the saved epoch and continues; with epochs=120
        # and a ckpt at ep18 it runs 19->120. Only imgsz/batch/device/epochs
        # survive as overrides — the rest revert to the checkpoint's args.
        resume=_resume_arg,
        batch=batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save_period=args.save_period,
        # Early stopping: fixes the "best at epoch ~20, then 200 wasted
        # declining epochs" pattern from NB88/89. EarlyStopping(patience)
        # halts when fitness hasn't improved for `patience` epochs; best.pt
        # already holds the peak.
        patience=args.patience,
        # CRITICAL: RMT-PPAD's vendored Ultralytics defaults val_period=50.
        # That gates BOTH validate() AND save_model() in trainer.py L529.
        # With our 30-epoch brief budget val never runs, trainer.metrics
        # stays empty, last.pt is never written, and our per-epoch
        # callbacks (drive_sync, metrics_table) have nothing to read.
        # Default 1 here so validation + save fire every epoch; pass
        # --val-period N on the CLI to ablate.
        val_period=args.val_period,
        lr0=lr0,
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        # P8 stability hardening after NaN-in-epoch-1: AMP off (FP16 can't
        # hold DETR's first-epoch loss_class_aux spikes ~400); longer
        # warmup; bias_lr=0 disables the SGD-era heuristic that AdamW
        # explodes under.
        amp=False,
        warmup_epochs=5.0,
        warmup_bias_lr=0.0,
        warmup_momentum=0.8,
        verbose=True,
        # deterministic=False: Ultralytics defaults to True, which calls
        # torch.use_deterministic_algorithms(True, warn_only=True). Several
        # ops we use have no deterministic CUDA kernel (the lane aux-seg
        # cross_entropy/nll_loss2d, CLRHead's grid_sampler ROIGather, our
        # histc score-histogram), so each emits a UserWarning every step and
        # falls back anyway. We don't need bit-exact reproducibility for
        # these experiments; turning it off silences all three warnings and
        # is marginally faster.
        deterministic=False,
        # Regularization to fight the detection overfitting (mAP peaked at
        # ep~20 then declined in NB88/89). The ROOT cause is that we had
        # ALL spatial aug off (the original RMT-PPAD trains with it ON);
        # weight_decay is a cheap orthogonal regularizer on top.
        weight_decay=args.weight_decay,
        # Horizontal flip is now mask-SAFE: RandomFlip was patched to also
        # flip the CLR lane_targets in sync (x->W-1-x, theta->1-theta), so
        # enabling fliplr no longer desyncs the curve supervision. This is
        # the single strongest regularizer and the main fix for overfitting.
        # The OTHER spatial transforms (scale/translate/mosaic/perspective)
        # stay OFF until their target transforms are implemented too.
        fliplr=args.fliplr,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # Photometric augmentation stays at defaults (safe for masks).
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4 from ultralytics defaults.
    )

    summary = {
        'mode': args.mode,
        'epochs': epochs,
        'batch': batch,
        'lr0': lr0,
        'name': args.name,
        'model_yaml': str(args.model_yaml),
        'data_yaml': str(args.data_yaml),
        'project': args.project,
    }
    (out_root / 'train_args.json').write_text(json.dumps(summary, indent=2))
    print('\n[train_lane_only] DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
