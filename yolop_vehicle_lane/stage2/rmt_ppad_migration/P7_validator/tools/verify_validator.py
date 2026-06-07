"""Phase P7 acceptance test.

Two checks:
  1. `lanes_to_mask` smoke (already in lane_rasterize.smoke_test).
  2. End-to-end: build clr_lane model, run an eval forward, call
     `MTDETRValidator.postprocess` on the result, assert the returned
     `mask` is a (B, 1, 640, 640) float tensor with at least some lane
     pixels (because we seeded high-confidence priors in cell 3, or
     because the random forward happened to fire at conf>0.4 anywhere).

We do NOT call a full `model.val(...)` here - that requires real BDD
images and the full metrics pipeline. P8 will do that. P7 just verifies
the postprocess + rasterizer plumb together correctly and produce the
shape the existing IoU/ACC code consumes.
"""
from __future__ import annotations

import argparse
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


class _DummyArgs:
    """Stands in for self.args in MTDETRValidator.postprocess."""
    imgsz = 640
    conf = 0.4
    mask_threshold = [0.45, 0.9]


class _DummyValidator:
    """Just enough of MTDETRValidator to call postprocess() with."""
    def __init__(self):
        import torch
        self.args = _DummyArgs()
        # mask_thr only consumed in the legacy (non-lane) branch; safe default.
        self.mask_thr = torch.tensor(self.args.mask_threshold).view(1, 2, 1, 1)


def smoke_test(rmt_ppad_root: Path, yaml_path: Path) -> int:
    import torch
    _ensure_vendor_first_on_path(rmt_ppad_root)

    print(f'[smoke] importing MTDETR from {rmt_ppad_root}', flush=True)
    from ultralytics import MTDETR  # noqa: E402
    from ultralytics.models.mtdetr.val import MTDETRValidator  # noqa: E402

    print(f'[smoke] building model from {yaml_path}', flush=True)
    model = MTDETR(str(yaml_path))
    inner = model.model.cpu().eval()
    print(f'[smoke] built; params={sum(p.numel() for p in inner.parameters()):,}', flush=True)

    fake_img = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = inner(fake_img)

    # In eval, MTDETRDecoder returns (y, x) where x is the full tuple.
    if isinstance(out, tuple) and len(out) == 2:
        y_pred, x_tuple = out
    else:
        print(f'[smoke] FAIL: unexpected forward output type {type(out).__name__}')
        return 1
    print(f'[smoke] forward returned y={tuple(y_pred.shape)} + x_tuple of len {len(x_tuple)}')

    # x_tuple = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, seg_mask, gate_mean)
    # The validator's postprocess consumes (preds[0], preds[1]) where preds[1] == x_tuple.
    seg_part = x_tuple[5]
    print(f'[smoke] seg_part type: {type(seg_part).__name__}')
    if not isinstance(seg_part, dict):
        print('[smoke] FAIL: seg_part should be dict (lane-only mode)')
        return 1
    if 'lane_output' not in seg_part:
        print(f'[smoke] FAIL: seg_part missing lane_output; keys={list(seg_part)}')
        return 1

    # Boost a few priors to high confidence so the conf_threshold doesn't
    # zero out everything (random init can land at low pos-score).
    lane_pred = seg_part['lane_output']
    if isinstance(lane_pred, (list, tuple)):
        lane_pred = lane_pred[-1]
    print(f'[smoke] lane prediction tensor shape: {tuple(lane_pred.shape)}')
    boosted = lane_pred.detach().clone()
    boosted[:, 0:3, 0] = -5.0   # neg low
    boosted[:, 0:3, 1] = 5.0    # pos high -> softmax(pos) ~ 1
    boosted[:, 0:3, 2] = 0.0    # start at bottom slot
    boosted[:, 0:3, 5] = 30.0   # 30-slot length
    # Make 3 obvious x columns at 100, 320, 500.
    import numpy as _np
    xs_a = _np.linspace(100, 130, 30) / 639.0
    xs_b = _np.linspace(320, 350, 30) / 639.0
    xs_c = _np.linspace(500, 530, 30) / 639.0
    boosted[:, 0, 6:6 + 30] = torch.from_numpy(xs_a.astype('float32'))
    boosted[:, 1, 6:6 + 30] = torch.from_numpy(xs_b.astype('float32'))
    boosted[:, 2, 6:6 + 30] = torch.from_numpy(xs_c.astype('float32'))

    new_x_tuple = list(x_tuple)
    new_x_tuple[5] = {'lane_output': boosted}
    preds_for_postprocess = [y_pred, tuple(new_x_tuple)]

    # Run postprocess.
    print('[smoke] running MTDETRValidator.postprocess on boosted predictions ...', flush=True)
    val = _DummyValidator()
    outputs, mask = MTDETRValidator.postprocess(val, preds_for_postprocess)
    print(f'[smoke] outputs: list of len {len(outputs)}; first shape: {tuple(outputs[0].shape)}')
    print(f'[smoke] mask shape: {tuple(mask.shape)}, dtype: {mask.dtype}')

    if tuple(mask.shape) != (2, 1, 640, 640):
        print(f'[smoke] FAIL: expected mask (2, 1, 640, 640), got {tuple(mask.shape)}')
        return 1
    n_lane_px = int(mask.sum())
    print(f'[smoke] mask total lane pixels (B=2): {n_lane_px}  (expect > 0)')
    if n_lane_px == 0:
        print('[smoke] FAIL: rasterized mask empty despite boosted priors')
        return 1

    print('\n[smoke] PASS')
    return 0


def _default_paths():
    here = Path(__file__).resolve()
    rmt_ppad_root = here.parent.parent.parent / 'vendor' / 'RMT-PPAD'
    yaml_path = (
        rmt_ppad_root / 'ultralytics' / 'cfg' / 'models' / 'mt-detr'
        / 'rtdetr-l_bdd_clr_lane.yaml'
    )
    return rmt_ppad_root, yaml_path


def main() -> int:
    default_rmt, default_yaml = _default_paths()
    p = argparse.ArgumentParser()
    p.add_argument('--rmt-ppad-root', type=Path, default=default_rmt)
    p.add_argument('--yaml', type=Path, default=default_yaml)
    args = p.parse_args()
    return smoke_test(args.rmt_ppad_root, args.yaml)


if __name__ == '__main__':
    sys.exit(main())
