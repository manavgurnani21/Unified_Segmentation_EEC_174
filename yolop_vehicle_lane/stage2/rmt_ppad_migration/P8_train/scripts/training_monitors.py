"""Training monitors for P8 ablation runs.

Two utilities, both designed to be installed before ultralytics' trainer
starts:

1. `install_tqdm_throttle()` patches `tqdm.tqdm.__init__` so all progress
   bars update at most every N iterations (default 100) or every M seconds
   (default 15). Cuts the per-batch log spam by ~100x while keeping a
   real-time-feel update cadence. Operates on the underlying tqdm class
   so it affects ultralytics' wrapped TQDM, dataset Scanning bars, etc.

2. `LossMonitor` + `make_callback()` give an ultralytics callback that
   tracks an EMA of each scalar in `trainer.loss_items` and stops training
   when any of these anomalies fires:
     - NaN / Inf in any loss value           (immediate stop)
     - sustained spike (>= spike_threshold *
       running EMA for spike_count consecutive batches)
     - LATE-PHASE divergence: after `warmup_batches` the EMA itself is
       trending UP for `divergence_window` consecutive ticks.

Why all of these? DETR-style models with SimOTA matching can briefly spike
when prior-to-GT assignment reshuffles (typically epochs 1-3). The
spike_threshold * spike_count combo lets those bounces through. Sustained
spikes or NaN are not normal and indicate optimizer state poisoning.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

# Names for our specific 3-element loss tuple (from MTDETRModel.loss()).
# Anything else falls back to generic indexed labels.
_KNOWN_LOSS_LABELS = {
    3: ['L_det', 'da_seg', 'll_seg'],
}


def install_tqdm_throttle(min_iters: int = 100, min_interval: float = 15.0) -> None:
    """Patch `tqdm.tqdm.__init__` so every progress bar throttles.

    Effect: with `nb=8750` batches/epoch and `min_iters=100`, you see about
    `8750 / 100 = 88` updates per epoch instead of 8750. Combined with the
    `min_interval=15.0` second floor, you also get sane behavior when
    iterations are fast (e.g. cache scan).

    Idempotent. Call once before ultralytics imports.
    """
    import tqdm as _tqdm

    if getattr(_tqdm.tqdm, '_p8_throttled', False):
        return  # already installed
    original_init = _tqdm.tqdm.__init__

    def _throttled_init(self, *args, **kwargs):
        kwargs.setdefault('miniters', min_iters)
        kwargs.setdefault('mininterval', min_interval)
        original_init(self, *args, **kwargs)

    _tqdm.tqdm.__init__ = _throttled_init
    _tqdm.tqdm._p8_throttled = True
    print(f'[training_monitors] tqdm throttled: miniters={min_iters} '
          f'mininterval={min_interval}s', flush=True)


class LossMonitor:
    """Per-batch monitor over the trainer's `loss_items` tensor.

    Watches each scalar with three independent checks:
      A. NaN/Inf check (fires immediately)
      B. Spike check: current > spike_threshold * EMA, sustained for
         spike_count consecutive batches
      C. Divergence check: after warmup_batches, the EMA itself climbing
         for divergence_window consecutive ticks (a slower failure mode)
    """

    def __init__(self,
                 ema_decay: float = 0.99,
                 spike_threshold: float = 5.0,
                 spike_count: int = 50,
                 warmup_batches: int = 200,
                 divergence_window: int = 200,
                 divergence_growth: float = 1.5,
                 log_every: int = 100,
                 ema_cap_mult: float = 10.0):
        """
        ema_cap_mult: per-batch values above `ema_cap_mult * EMA` are
            still counted as spike events (via the consecutive_spikes
            counter) but their contribution to the EMA blend is capped
            at that multiple. Prevents one outlier batch from pulling
            the EMA way up and then taking hundreds of batches to decay
            back, which falsely trips the divergence check. Set to 0
            or None to disable.
        """
        self.ema_decay = ema_decay
        self.spike_threshold = spike_threshold
        self.spike_count = spike_count
        self.warmup_batches = warmup_batches
        self.divergence_window = divergence_window
        self.divergence_growth = divergence_growth
        self.log_every = log_every
        self.ema_cap_mult = ema_cap_mult

        # State
        self._step = 0
        self.ema: Dict[str, float] = {}
        self.consecutive_spikes: Dict[str, int] = {}
        self.ema_history: Dict[str, List[float]] = {}
        self.last_log_step = 0

    def _label_for(self, items_len: int) -> List[str]:
        return _KNOWN_LOSS_LABELS.get(items_len,
                                      [f'loss[{i}]' for i in range(items_len)])

    def update(self, losses: Dict[str, float]) -> Tuple[Optional[str], bool]:
        """Returns (info_msg or None, should_stop_bool)."""
        self._step += 1
        critical_msgs = []
        info_msgs = []

        for k, v in losses.items():
            if v is None:
                continue

            # A. NaN/Inf -> immediate stop
            if math.isnan(v) or math.isinf(v):
                return (f'step {self._step}: {k} = {v!r} (NaN/Inf)', True)

            # Seed EMA on first observation.
            if k not in self.ema:
                self.ema[k] = v
                self.consecutive_spikes[k] = 0
                self.ema_history[k] = [v]
                continue

            ema = self.ema[k]

            # B. Spike check.
            if ema > 1e-6 and v > self.spike_threshold * ema:
                self.consecutive_spikes[k] += 1
                if self.consecutive_spikes[k] >= self.spike_count:
                    critical_msgs.append(
                        f'{k} sustained spike: current={v:.3g} ema={ema:.3g} '
                        f'(>{self.spike_threshold}x EMA for {self.consecutive_spikes[k]} batches)'
                    )
                elif self.consecutive_spikes[k] in (1, 10, 25):
                    info_msgs.append(
                        f'{k} spike: current={v:.3g} ema={ema:.3g} '
                        f'(consec={self.consecutive_spikes[k]})'
                    )
            else:
                self.consecutive_spikes[k] = 0

            # Update EMA AFTER comparing to it so the comparison is fair.
            # P8 iter-1 fix: cap the value used in the EMA blend so a single
            # transient outlier (e.g. one bad batch contributing ll_seg=6000
            # while the running average is 3) can't poison the EMA and
            # falsely trip the divergence check ~hundreds of batches later.
            # The spike check above already tracks the raw value for the
            # consecutive_spikes / sustained-spike logic.
            if self.ema_cap_mult and self.ema_cap_mult > 0 and ema > 1e-6:
                v_for_ema = min(v, self.ema_cap_mult * ema)
            else:
                v_for_ema = v
            self.ema[k] = self.ema_decay * ema + (1 - self.ema_decay) * v_for_ema

            # C. Divergence: EMA climbing over a window. We snapshot once per
            # log_every steps to keep history small.
            if self._step >= self.warmup_batches and self._step % self.log_every == 0:
                hist = self.ema_history[k]
                hist.append(self.ema[k])
                if len(hist) > self.divergence_window // max(1, self.log_every) + 5:
                    hist.pop(0)
                # If the most recent EMA is much larger than the EMA from
                # divergence_window/log_every snapshots ago: trouble.
                tail_n = self.divergence_window // max(1, self.log_every)
                if len(hist) >= tail_n + 1:
                    old, cur = hist[-tail_n - 1], hist[-1]
                    if old > 1e-6 and cur > self.divergence_growth * old:
                        critical_msgs.append(
                            f'{k} divergence: EMA {old:.3g} -> {cur:.3g} '
                            f'over last {self.divergence_window} batches'
                        )

        if critical_msgs:
            return (' | '.join(critical_msgs), True)
        if info_msgs and (self._step - self.last_log_step) >= self.log_every:
            self.last_log_step = self._step
            return ('warn: ' + ' | '.join(info_msgs), False)
        return (None, False)


def make_callback(monitor: Optional[LossMonitor] = None,
                  also_print_every: int = 0) -> 'tuple':
    """Returns a `(name, callable)` pair to register with
    `model.add_callback('on_train_batch_end', callback)`.

    If `also_print_every > 0`, additionally print a one-line per-key EMA
    summary every N batches (helps debug what the EMA actually thinks).
    """
    mon = monitor if monitor is not None else LossMonitor()

    def _cb(trainer):
        items = getattr(trainer, 'loss_items', None)
        if items is None:
            return
        try:
            flat = items.detach().cpu().flatten().tolist()
        except AttributeError:
            try:
                flat = list(items)
            except TypeError:
                return
        labels = mon._label_for(len(flat))
        losses = {label: float(v) for label, v in zip(labels, flat)}
        msg, should_stop = mon.update(losses)
        if msg:
            print(f'\n[loss-monitor] step={mon._step} {msg}', flush=True)
        if also_print_every > 0 and mon._step % also_print_every == 0:
            ema_str = ' '.join(f'{k}={mon.ema.get(k, float("nan")):.3g}' for k in labels)
            print(f'[loss-monitor] ema: {ema_str}', flush=True)
        if should_stop:
            print(f'\n[loss-monitor] STOPPING TRAINING NOW: {msg}', flush=True)
            # Two-pronged stop: signal the trainer's stopper AND raise.
            try:
                trainer.stop = True
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f'training stopped by LossMonitor: {msg}')

    return ('on_train_batch_end', _cb)


__all__ = [
    'install_tqdm_throttle',
    'LossMonitor',
    'make_callback',
]
