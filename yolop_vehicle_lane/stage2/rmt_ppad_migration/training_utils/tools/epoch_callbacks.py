"""Per-epoch Drive sync + metrics-table callbacks.

Two callables, both registered via `model.add_callback('on_fit_epoch_end', cb)`
so they fire AFTER validation has populated `trainer.metrics`. Both are
self-contained and safe to call from multiple training scripts.

Drive sync:
    At the end of every epoch (after Ultralytics writes last.pt / best.pt
    locally to /content/runs/<name>/weights/), mirror the following to
    /content/drive/MyDrive/EcoCAR/training_runs/checkpoints/<run_name>/:

        last.pt                    - always (epoch end)
        best.pt                    - only when its mtime changed (= val improved)
        results.csv                - always (Ultralytics appends one row per epoch)
        log_epoch_NNN.json         - new per-epoch metrics snapshot

    Designed to fail open: a Drive copy error logs a warning but does
    NOT stop training (Drive can be slow / occasionally I/O-bound;
    don't blow up a 30h run for a transient hiccup).

Metrics table:
    Reads `trainer.metrics` (populated by the validator's get_stats()
    via the val.py patch above) and prints a fixed-width row with both
    detection and lane keys. Header printed once at the top.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Full-log tee
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate writes to the original stream AND a line-buffered file.

    Installed on sys.stdout/sys.stderr by `install_full_log_tee` so the
    ENTIRE training run (model summary, the '[MTDETRDLoss] lane matching
    scheme = ...' line, every warning, every epoch row) is captured to a
    file the drive_sync callback mirrors to Drive each epoch. The console
    still gets everything (run_streaming keeps working).
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
            self._fh.flush()  # line-by-line so a Colab disconnect loses nothing
        except Exception:  # noqa: BLE001
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_full_log_tee(log_path) -> None:
    """Tee stdout+stderr into `log_path` (APPEND, line-buffered).

    APPEND ('a') not truncate ('w'): the old 'w' wiped the log on every
    resume, so a multi-session run (250 epochs cannot finish in one Colab
    session) ended up with a log showing ONLY the last partial session -
    the "incomplete log" bug. With 'a', resumes (FRESH=False) extend the
    same log. A genuinely fresh run (FRESH=True) deletes the whole run dir
    in the notebook first, so the log also starts clean. A divider marks
    each (re)attach so sessions are distinguishable.

    The file handle is intentionally leaked for the process lifetime
    (closed on exit) so all late output is captured.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'a', encoding='utf-8', buffering=1)
    fh.write('\n' + '=' * 70 + '\n[full-log] session (re)start\n' + '=' * 70 + '\n')
    fh.flush()
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f'[full-log] teeing stdout+stderr (append) -> {log_path}', flush=True)


# ---------------------------------------------------------------------------
# Drive sync
# ---------------------------------------------------------------------------

def make_drive_sync_callback(
    drive_checkpoint_root: Path,
    run_name: str,
    sync_results_csv: bool = True,
    write_log_json: bool = True,
) -> Tuple[str, callable]:
    """Build an `on_fit_epoch_end` Drive-sync callback.

    Args:
        drive_checkpoint_root: parent directory on Drive that hosts
            per-run subdirs (e.g.
            `/content/drive/MyDrive/EcoCAR/training_runs/checkpoints`).
        run_name: subdirectory name for this run (e.g. 'clr_lane_default').
        sync_results_csv: also mirror Ultralytics' results.csv at end of
            every epoch. Cheap (a few KB).
        write_log_json: write a per-epoch `log_epoch_NNN.json` snapshot
            with all of `trainer.metrics`. Useful for grep-based
            inspection of historical epochs.
    """
    drive_root = Path(drive_checkpoint_root)
    drive_dir = drive_root / run_name
    drive_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'last_best_mtime': 0.0,
        'epochs_synced': 0,
    }

    def _safe_copy(src: Path, dst: Path) -> Tuple[bool, Optional[str]]:
        try:
            if not src.exists():
                return False, 'src missing'
            shutil.copy2(src, dst)
            return True, None
        except Exception as e:  # noqa: BLE001
            return False, f'{type(e).__name__}: {e}'

    def _cb(trainer):
        epoch = getattr(trainer, 'epoch', None)
        if epoch is None:
            return

        # Prefer trainer.last / trainer.best (the EXACT paths Ultralytics
        # writes to). Fall back to <save_dir>/weights/{last,best}.pt only
        # if those attributes are missing.
        save_dir = Path(getattr(trainer, 'save_dir', '.'))
        weights_dir = save_dir / 'weights'
        t0 = time.time()

        # 1. last.pt (always, end of every epoch)
        last_pt_src = Path(getattr(trainer, 'last', None) or (weights_dir / 'last.pt'))
        last_ok, last_err = _safe_copy(last_pt_src, drive_dir / 'last.pt')

        # 2. best.pt (only when its mtime changed since the previous sync,
        # which == Ultralytics rewrote it because val improved)
        best_pt_src = Path(getattr(trainer, 'best', None) or (weights_dir / 'best.pt'))
        best_ok = False
        best_err = None
        best_changed = False
        if best_pt_src.exists():
            mtime = best_pt_src.stat().st_mtime
            if mtime > state['last_best_mtime']:
                best_ok, best_err = _safe_copy(best_pt_src, drive_dir / 'best.pt')
                if best_ok:
                    state['last_best_mtime'] = mtime
                    best_changed = True

        # 3. results.csv (cheap, always)
        if sync_results_csv:
            _safe_copy(save_dir / 'results.csv', drive_dir / 'results.csv')

        # 3b. FULL training log (every epoch). The train script tees all
        # stdout+stderr to <save_dir>/full_train.log (line-buffered); we
        # mirror it to Drive each epoch so the construction-time output
        # (e.g. '[MTDETRDLoss] lane matching scheme = hungarian', the
        # model summary, every warning) and the complete per-epoch detail
        # survive even if Colab disconnects or the run_streaming notebook
        # log fails to flush. This is what lets us answer "was Hungarian
        # actually active?" after the fact.
        for log_name in ('full_train.log', 'train.log'):
            src = save_dir / log_name
            if src.exists():
                _safe_copy(src, drive_dir / log_name)
                break

        # 4. per-epoch metrics snapshot
        if write_log_json:
            metrics_raw = getattr(trainer, 'metrics', {}) or {}
            metrics = {}
            for k, v in metrics_raw.items():
                if hasattr(v, 'item'):
                    try:
                        metrics[k] = float(v.item())
                        continue
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    metrics[k] = str(v)

            log = {
                'epoch': epoch + 1,
                'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'fitness': float(getattr(trainer, 'fitness', 0.0) or 0.0),
                'best_fitness': float(getattr(trainer, 'best_fitness', 0.0) or 0.0),
                'best_pt_updated_this_epoch': best_changed,
                'metrics': metrics,
            }
            # Include training loss tensor if available
            loss_items = getattr(trainer, 'loss_items', None)
            if loss_items is not None:
                try:
                    log['loss_items'] = loss_items.detach().cpu().flatten().tolist()
                except Exception:  # noqa: BLE001
                    pass
            log_path = drive_dir / f'log_epoch_{epoch + 1:03d}.json'
            try:
                log_path.write_text(json.dumps(log, indent=2))
            except Exception as e:  # noqa: BLE001
                print(f'[drive_sync] log JSON write failed: {e}', flush=True)

        elapsed = time.time() - t0
        state['epochs_synced'] += 1
        msg = (
            f'[drive_sync] epoch={epoch + 1} '
            f'last.pt={"ok" if last_ok else f"FAIL({last_err})"} '
            f'best.pt={"updated" if best_changed else ("ok" if best_ok else "unchanged")} '
            f'sync={elapsed:.1f}s'
        )
        print(msg, flush=True)

    return ('on_fit_epoch_end', _cb)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

# Default set of metric keys to surface in the per-epoch table.
# `(label, list_of_candidate_keys)` - the first key that exists in
# `trainer.metrics` wins, so this works across renames between Ultralytics
# versions.
_DEFAULT_METRIC_COLUMNS = [
    ('ep',         []),  # filled from trainer.epoch
    ('P(det)',     ['metrics/precision(B)']),
    ('R(det)',     ['metrics/recall(B)']),
    ('mAP50',      ['metrics/mAP50(B)']),
    ('mAP50-95',   ['metrics/mAP50-95(B)']),
    ('IoU(lane)',  ['metrics/IoU(lane)']),
    ('F1(lane)',   ['metrics/lane_f1(lane)']),  # curve F1@0.5 (0 until lanes are sharp)
    ('curveIoU',   ['metrics/lane_curveIoU(lane)']),  # mean best-match IoU (un-saturated; watch THIS)
    ('mIoU(lane)', ['metrics/mIoU(lane)']),
    ('pixAcc',     ['metrics/pixacc(lane)']),
    ('subAcc',     ['metrics/subacc(lane)']),
    # Phase 1: lane cls-head liveness columns. score_std/spread near 0 =
    # collapsed/dead cls head (the frozen-IoU signature).
    ('scoreStd',   ['metrics/lane_score_std(lane)']),
    ('scoreSprd',  ['metrics/lane_score_spread(lane)']),
]


def make_metrics_table_callback(
    columns=_DEFAULT_METRIC_COLUMNS,
    col_width: int = 11,
    dead_lane_std: float = 1e-3,
    dead_lane_patience: int = 5,
) -> Tuple[str, callable]:
    """Build an `on_fit_epoch_end` callback that prints a per-epoch
    fixed-width row of detection + lane metrics AND auto-detects a dead
    lane cls head.

    The header is printed exactly once (at the first epoch end). Missing
    keys are rendered as `-`.

    Dead-lane monitor: if `metrics/lane_score_std(lane)` stays below
    `dead_lane_std` for `dead_lane_patience` consecutive validated
    epochs, a loud warning is printed. A collapsed cls (std ~ 0) is the
    root cause of frozen decoded IoU; surfacing it automatically means we
    don't have to eyeball the histogram every run.
    """
    state = {'header_printed': False, 'dead_streak': 0, 'iou_prev': None,
             'iou_frozen_streak': 0}

    def _fmt(v) -> str:
        if v is None:
            return '-'.rjust(col_width)
        if isinstance(v, float):
            return f'{v:.4f}'.rjust(col_width)
        return str(v).rjust(col_width)

    def _get(metrics, keys):
        for k in keys:
            if k in metrics:
                return metrics[k]
        return None

    def _cb(trainer):
        metrics = getattr(trainer, 'metrics', None) or {}
        epoch = getattr(trainer, 'epoch', 0)

        # Reprint the header EVERY epoch. The old behavior printed it once,
        # so epoch 2+ showed a bare unlabeled number row buried among the
        # per-epoch [lane-score]/[lane-geom]/monitor lines - unreadable.
        # A 2-line header per epoch is cheap and makes each row self-
        # contained and aligned with its columns.
        header = ''.join(label.rjust(col_width) for label, _ in columns)
        sep = ''.join(('-' * (col_width - 1) + ' ') for _ in columns)
        print('\n' + header, flush=True)
        print(sep, flush=True)

        cells = []
        for label, candidate_keys in columns:
            if label == 'ep':
                cells.append(f'{epoch + 1}'.rjust(col_width))
                continue
            cells.append(_fmt(_get(metrics, candidate_keys)))
        print(''.join(cells), flush=True)

        # --- dead-lane auto-check (the "loss monitor" for the lane head) ---
        std = _get(metrics, ['metrics/lane_score_std(lane)'])
        if std is not None:
            try:
                std = float(std)
            except (TypeError, ValueError):
                std = None
        if std is not None:
            if std < dead_lane_std:
                state['dead_streak'] += 1
            else:
                state['dead_streak'] = 0
            # Fire ONCE when first crossing the threshold, then re-arm only
            # after the condition clears (dead_streak reset to 0). Avoids
            # the 100+ identical warnings/run seen in NB95.
            if state['dead_streak'] == dead_lane_patience:
                print(
                    f'\n[dead-lane-monitor] *** LANE CLS HEAD APPEARS DEAD *** '
                    f'pos-score std={std:.6f} < {dead_lane_std} for '
                    f'{state["dead_streak"]} consecutive val epochs. The cls '
                    f'has collapsed to a uniform sigmoid; decoded IoU cannot '
                    f'learn. Check the matching scheme (LANE_MATCH=hungarian?). '
                    f'(this warning fires once; re-arms if std recovers)',
                    flush=True,
                )

        # PRIMARY frozen-IoU monitor. This is the signal that actually
        # matters and the one the std-based check above MISSES: decoded
        # IoU can be byte-frozen even while the cls score_std is healthy
        # and rising (observed in NB88/89: score_std 0.069 -> 0.084 while
        # IoU stuck at exactly 0.0718). That combination means the cls is
        # learning but the GEOMETRY (curve shape) is not - the rasterized
        # mask is constant regardless of which priors the scores select.
        # The std-monitor can't see this because it only watches the cls.
        # Patience is short (3) and the message is loud so this surfaces
        # fast instead of needing 6 epochs like the old "note".
        iou = _get(metrics, ['metrics/IoU(lane)'])
        try:
            iou = float(iou) if iou is not None else None
        except (TypeError, ValueError):
            iou = None
        if iou is not None:
            if state['iou_prev'] is not None and abs(iou - state['iou_prev']) < 1e-6:
                state['iou_frozen_streak'] += 1
            else:
                state['iou_frozen_streak'] = 0
            state['iou_prev'] = iou
            # Fire ONCE on first crossing patience; re-arms after IoU moves.
            if state['iou_frozen_streak'] == 3:
                std_txt = f'{std:.5f}' if std is not None else 'n/a'
                print(
                    f'\n[frozen-iou-monitor] *** DECODED IoU(lane) FROZEN *** '
                    f'at {iou:.4f} for {state["iou_frozen_streak"]} consecutive '
                    f'val epochs while cls score_std={std_txt}. If score_std is '
                    f'healthy (>0.01) the CLS is learning but the GEOMETRY is '
                    f'not - the predicted curves are stuck at their anchor '
                    f'positions, so the rasterized mask never changes. Check '
                    f'the lane regression branch (reg_layers / xytl+iou loss '
                    f'weights / diff-clamp), not the matching scheme.',
                    flush=True,
                )

    return ('on_fit_epoch_end', _cb)


__all__ = [
    'install_full_log_tee',
    'make_drive_sync_callback',
    'make_metrics_table_callback',
]
