"""Multi-task-learning anti-negative-transfer toolkit (CLI-togglable).

Central home for the five MTL defenses we ablate against the lane-IoU
"peak then decline" negative transfer. Keeping the *logic* here (with pure,
unit-testable cores) lets the vendored Ultralytics trainer carry only a
tiny, default-OFF hook for each one.

The five techniques and how each plugs in:

  1. PCGrad (gradient surgery)        -> `pcgrad_backward_and_set_grad`,
       called from trainer's train loop when env MTL_PCGRAD=1.
  2. Asymmetric LR (backbone x mult)  -> `install_asymmetric_lr`,
       monkeypatches MTDETRTrainer.build_optimizer (train script).
  3. Strengthen GCA decoupling        -> `set_seg_gate_floor`,
       raises LiteDynamicGate.clamp_min on the lane path (train script).
  4. Dynamic det-loss decay           -> `make_det_decay_callback` +
       `get_det_loss_scale`; callback sets env DET_LOSS_SCALE at epoch N,
       tasks.py multiplies L_det by it.
  5. Asymmetric trunk freeze          -> already in train_lane_only.py
       (`--freeze-trunk-after`); listed here for completeness.

The task-loss list produced by the model is [L_det, da_seg, ll_seg]
(detection=index 0, drivable=1 (=0 in lane-only), lane=index 2).

PCGrad NOTE: requires AMP off (amp=False), which is the standing config
for this project (FP16 can't hold the first-epoch DETR spikes). With AMP
on, the GradScaler would scale the naive .grad but not the autograd.grad
per-task grads, producing a scale mismatch. The trainer hook guards this.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Shared: identify backbone+neck parameters (mirrors trainer.get_backbone_params)
# ---------------------------------------------------------------------------

def iter_backbone_named_params(model, boundary_index: int = 27):
    """Yield (name, param) for params in model.model.<idx> with idx<=boundary.

    Matches the EXACT selection rule of the vendored trainer's
    `get_backbone_params` so asym-LR, PCGrad and freeze all agree on what
    "the shared trunk" is. `model` is the DetectionModel whose `.model` is
    the nn.Sequential of layers; param names look like 'model.7.conv.weight'.
    """
    for name, param in model.named_parameters():
        if "model." in name:
            parts = name.split(".")
            if len(parts) > 2 and parts[1].isdigit() and int(parts[1]) <= boundary_index:
                yield name, param


# ---------------------------------------------------------------------------
# 1. PCGrad — projecting conflicting gradients
# ---------------------------------------------------------------------------

def pcgrad_project_vector(det_vec, lane_vec, eps: float = 1e-12):
    """PURE core: project the detection gradient off the lane gradient when
    they conflict (asymmetric PCGrad that PROTECTS the lane task).

    If <det, lane> < 0 the two tasks disagree on the shared trunk; remove
    det's lane-conflicting component:
        det_proj = det - (<det,lane> / |lane|^2) * lane
    Otherwise det is left unchanged. Returns (det_out, stats) where stats =
    {cos, conflict, dot}. Operates on the flattened global gradient (the
    PCGrad paper treats each task's whole gradient as one vector).
    """
    import torch

    dot = torch.dot(det_vec, lane_vec)
    dn = det_vec.norm()
    ln = lane_vec.norm()
    cos = float(dot / (dn * ln + eps))
    conflict = bool(dot.item() < 0.0)
    if conflict:
        det_out = det_vec - (dot / (ln * ln + eps)) * lane_vec
    else:
        det_out = det_vec
    return det_out, {"cos": cos, "conflict": conflict, "dot": float(dot)}


def _flatten_grads(params, grads):
    """Flatten a per-param grad tuple to one vector, zero-filling None
    (allow_unused). Returns (vector, shapes, numels)."""
    import torch

    flat, shapes, numels = [], [], []
    for p, g in zip(params, grads):
        gi = g if g is not None else torch.zeros_like(p)
        flat.append(gi.reshape(-1))
        shapes.append(p.shape)
        numels.append(p.numel())
    return torch.cat(flat), shapes, numels


def pcgrad_backward_and_set_grad(
    loss_list: Sequence,
    backbone_params: List,
    det_idx: int = 0,
    lane_idx: int = 2,
):
    """Run the PCGrad backward for one step.

    1. naive `sum(loss).backward(retain_graph=True)` -> ACCUMULATES the
       correct .grad for every param (heads get only their own task; the
       backbone gets the naive det+lane sum). Using `.backward()` (+=)
       preserves gradient accumulation over micro-batches.
    2. per-task backbone grads via autograd.grad (re-traverses retained
       graph): det then lane.
    3. project det off lane on the flattened backbone vector when conflicting.
    4. apply the projection DELTA `(det_proj - det)` to the backbone .grad
       as a `+=` CORRECTION. Net backbone grad becomes (det_proj + lane);
       heads are untouched; accumulation still works because we add rather
       than overwrite, and a no-conflict step contributes exactly zero.

    Returns (scalar_loss_detached, stats). Caller then runs optimizer_step.
    Assumes AMP off (see module docstring).
    """
    import torch

    total = sum(loss_list)
    total.backward(retain_graph=True)  # accumulate naive grads (heads + backbone)

    det_grads = torch.autograd.grad(
        loss_list[det_idx], backbone_params, retain_graph=True, allow_unused=True,
    )
    lane_grads = torch.autograd.grad(
        loss_list[lane_idx], backbone_params, retain_graph=False, allow_unused=True,
    )

    det_v, shapes, numels = _flatten_grads(backbone_params, det_grads)
    lane_v, _, _ = _flatten_grads(backbone_params, lane_grads)

    det_proj, stats = pcgrad_project_vector(det_v, lane_v)
    correction = det_proj - det_v  # exactly zero when no conflict

    # Add the projection correction onto the (already-accumulated) backbone
    # grad slot-by-slot. backbone naive grad was (det+lane); + (det_proj-det)
    # = (det_proj + lane), the PCGrad-resolved update, without disturbing
    # accumulation or the head gradients.
    off = 0
    for p, shp, n in zip(backbone_params, shapes, numels):
        delta = correction[off:off + n].view(shp)
        if p.grad is None:
            p.grad = delta.clone()
        else:
            p.grad = p.grad + delta
        off += n
    return total.detach(), stats


# ---------------------------------------------------------------------------
# 2. Asymmetric LR — backbone group at base_lr * mult
# ---------------------------------------------------------------------------

def split_backbone_param_group(optimizer, model, boundary_index: int, mult: float,
                               logger: Callable = print) -> int:
    """Move backbone+neck params into NEW optimizer groups at lr*mult.

    Must run BEFORE the LR scheduler is constructed (LambdaLR snapshots
    base_lrs from the live param_groups). Called inside the wrapped
    build_optimizer, which returns just before `_setup_scheduler()`.

    Returns the number of backbone params relocated.
    """
    bb_ids = {id(p) for _n, p in iter_backbone_named_params(model, boundary_index)}
    if not bb_ids:
        logger("[asym-lr] WARNING: no backbone params matched - nothing scaled")
        return 0
    moved = 0
    for g in list(optimizer.param_groups):
        bb = [p for p in g["params"] if id(p) in bb_ids]
        if not bb:
            continue
        rest = [p for p in g["params"] if id(p) not in bb_ids]
        g["params"] = rest
        new_g = {k: v for k, v in g.items() if k != "params"}
        new_g["params"] = bb
        new_g["lr"] = g.get("lr", 0.0) * mult
        optimizer.add_param_group(new_g)
        moved += len(bb)
    logger(f"[asym-lr] moved {moved} backbone params (layers 0..{boundary_index}) "
           f"to lr*{mult}; optimizer now has {len(optimizer.param_groups)} groups")
    return moved


def install_asymmetric_lr(trainer_cls, boundary_index: int, mult: float,
                          logger: Callable = print) -> None:
    """Monkeypatch trainer_cls.build_optimizer to split-and-scale the
    backbone group. Idempotent (won't double-wrap)."""
    if getattr(trainer_cls, "_asym_lr_installed", False):
        return
    orig = trainer_cls.build_optimizer

    def wrapped(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        opt = orig(self, model, name=name, lr=lr, momentum=momentum,
                   decay=decay, iterations=iterations)
        try:
            split_backbone_param_group(opt, model, boundary_index, mult, logger)
        except Exception as e:  # noqa: BLE001 - never break optimizer build
            logger(f"[asym-lr] WARNING: split failed, using uniform lr: {e}")
        return opt

    trainer_cls.build_optimizer = wrapped
    trainer_cls._asym_lr_installed = True
    logger(f"[asym-lr] armed: backbone(layers 0..{boundary_index}) lr*{mult}")


# ---------------------------------------------------------------------------
# 3. Strengthen GCA decoupling — raise the lane gate's clamp floor
# ---------------------------------------------------------------------------

def set_seg_gate_floor(model, floor: float, logger: Callable = print) -> int:
    """Raise LiteDynamicGate.clamp_min on the lane/seg path.

    GCA fuses `out = shared + gate*(task - shared)`; a higher gate floor
    forces the lane branch to inject MORE of its own task-specific
    transform (less raw shared-trunk passthrough), directly countering the
    diagnosed "Seg_gate stuck ~0.54 => lane stays coupled to the detection-
    dominated trunk". Targets gates whose owner attribute name hints seg/
    lane; if none match, falls back to ALL gates and says so.

    Returns the number of gates adjusted.
    """
    seg_like, all_gates = [], []
    for mod_name, module in model.named_modules():
        if type(module).__name__ == "LiteDynamicGate":
            all_gates.append((mod_name, module))
            if any(tok in mod_name.lower() for tok in ("seg", "lane", "ll")):
                seg_like.append((mod_name, module))
    targets = seg_like or all_gates
    if not targets:
        logger("[gca-floor] WARNING: no LiteDynamicGate found - nothing changed")
        return 0
    if not seg_like:
        logger("[gca-floor] note: couldn't name-match a seg/lane gate; "
               "applying floor to ALL gates")
    for mod_name, module in targets:
        old = getattr(module, "clamp_min", None)
        module.clamp_min = float(floor)
        logger(f"[gca-floor] {mod_name}.clamp_min {old} -> {floor}")
    return len(targets)


# ---------------------------------------------------------------------------
# 4. Dynamic det-loss decay — env-driven scale on L_det
# ---------------------------------------------------------------------------

_DET_SCALE_ENV = "DET_LOSS_SCALE"


def get_det_loss_scale() -> float:
    """Read the current detection-loss scale (1.0 = unchanged). tasks.py
    calls this each forward and multiplies L_det by it."""
    try:
        return float(os.environ.get(_DET_SCALE_ENV, "1.0"))
    except (TypeError, ValueError):
        return 1.0


def make_det_decay_callback(decay_epoch: int, factor: float,
                            logger: Callable = print) -> Tuple[str, Callable]:
    """Return (hook_name, callback) that, at `decay_epoch`, sets the env
    det-loss scale to `factor` (e.g. 0.5) once. Detection converges fast
    and then dominates the shared trunk; halving its loss hands gradient
    priority back to the lane head for the rest of training.

    WARNING: prior exp01-71 found moving weight off detection can hurt det
    mAP - this is an ablation knob, not a default.
    """
    state = {"done": False}

    def _cb(trainer, _st=state, _e=decay_epoch, _f=factor):
        if _st["done"]:
            return
        if int(getattr(trainer, "epoch", 0)) >= _e:
            os.environ[_DET_SCALE_ENV] = str(_f)
            _st["done"] = True
            logger(f"\n[det-decay] epoch {trainer.epoch}: DET_LOSS_SCALE -> {_f} "
                   f"(handing gradient priority to the lane head)")

    return "on_train_epoch_start", _cb


def reset_det_loss_scale() -> None:
    """Clear the env scale (call at train start so a re-used process doesn't
    inherit a stale decay)."""
    os.environ.pop(_DET_SCALE_ENV, None)


__all__ = [
    "iter_backbone_named_params",
    "pcgrad_project_vector", "pcgrad_backward_and_set_grad",
    "split_backbone_param_group", "install_asymmetric_lr",
    "set_seg_gate_floor",
    "get_det_loss_scale", "make_det_decay_callback", "reset_det_loss_scale",
]
