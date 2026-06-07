"""Pretrained backbone loader with shape-mismatch tolerance.

Loads any .pth/.pt checkpoint into the joint model's backbone with the following
behaviour:

1. Unwraps common checkpoint formats (raw state_dict, {'model': ...},
   {'state_dict': ...}, {'ema_state_dict': ...}, lightning ckpts).
2. Strips common prefixes (`module.`, `backbone.`, `_orig_mod.`) so weights
   trained as `module.backbone.x` or `_orig_mod.backbone.x` match our
   `backbone.x` parameter names.
3. Skips any tensor whose shape doesn't match the target parameter (with a log
   line), instead of erroring -- so we can use a YOLOP / YOLOPv2 / CULane-
   pretrained checkpoint as a partial init even when only some layers map.
4. Returns (loaded_count, total_target_count) so the caller can check coverage.

Why we need this: through 40 experiments the backbone was always randomly
initialized. NB45 (30 epochs, width=1.0) was the only run where det reached
val_map50 = 0.011 -- everything else stuck near zero -- precisely because
random-init backbones take 5-10x longer to converge on detection. A real
pretrained init is the single biggest lever left untouched.
"""
from __future__ import annotations

from typing import Tuple

import torch


def _unwrap_checkpoint(obj):
    """Find the actual state_dict inside a torch.load output."""
    if isinstance(obj, dict):
        for key in ('state_dict', 'model', 'ema_state_dict', 'weights', 'net'):
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                if any(isinstance(v, torch.Tensor) for v in inner.values()):
                    return inner
                # Sometimes nested: {'model': {'state_dict': ...}}
                inner2 = _unwrap_checkpoint(inner)
                if inner2 is not None:
                    return inner2
        # If the dict itself looks like a state_dict (all values are tensors).
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    return None


_STRIP_PREFIXES = (
    'module.',
    '_orig_mod.',
    'model.',
    'net.',
    'backbone.',  # Match a 'backbone.foo' source key to a 'foo' target name -- last to ensure other prefixes are stripped first.
)


def _normalize_keys(state_dict):
    """Return a new dict where common wrapping prefixes are stripped."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        # Strip top-level wrappers like 'module.' or '_orig_mod.' first.
        for prefix in ('module.', '_orig_mod.'):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        out[nk] = v
    return out


def load_pretrained_backbone(model, checkpoint_path: str) -> Tuple[int, int]:
    """Load matching parameters from `checkpoint_path` into model.backbone.

    Returns (loaded_count, total_backbone_tensors).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg.
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = _unwrap_checkpoint(ckpt)
    if state is None:
        raise RuntimeError(f'Could not find a state_dict inside {checkpoint_path}')
    state = _normalize_keys(state)

    backbone = getattr(model, 'backbone', None)
    if backbone is None and hasattr(model, '_orig_mod'):
        backbone = getattr(model._orig_mod, 'backbone', None)
    if backbone is None:
        raise RuntimeError('Model has no .backbone attribute to load into')

    target_sd = backbone.state_dict()
    loaded = 0
    skipped_shape = 0
    skipped_missing = 0
    matched_keys = []
    for target_name, target_tensor in target_sd.items():
        # Try direct match first, then `backbone.<name>`, then strip-`backbone.` variants.
        candidate_keys = [
            target_name,
            f'backbone.{target_name}',
        ]
        # If target name itself starts with 'backbone.', try without.
        if target_name.startswith('backbone.'):
            candidate_keys.append(target_name[len('backbone.'):])
        src_tensor = None
        chosen_key = None
        for k in candidate_keys:
            if k in state:
                src_tensor = state[k]
                chosen_key = k
                break
        if src_tensor is None:
            skipped_missing += 1
            continue
        if src_tensor.shape != target_tensor.shape:
            skipped_shape += 1
            continue
        target_sd[target_name] = src_tensor.to(target_tensor.dtype)
        matched_keys.append((chosen_key, target_name))
        loaded += 1
    backbone.load_state_dict(target_sd, strict=False)
    if loaded == 0:
        print(f'[pretrained_loader] WARNING: 0/{len(target_sd)} tensors matched. '
              f'Total source keys: {len(state)}. Sample source keys: '
              f'{list(state.keys())[:5]}. Sample target keys: '
              f'{list(target_sd.keys())[:5]}')
    else:
        print(f'[pretrained_loader] matched {loaded}/{len(target_sd)} '
              f'(skipped {skipped_missing} missing, {skipped_shape} shape mismatch)')
    return loaded, len(target_sd)
