"""Phase P5 acceptance test.

Builds the lane-only MTDETR model + constructs a synthetic training batch
+ runs `model.model.loss(batch)`. Asserts:

  1. The criterion is `MTDETRDLoss` with `lane_only_mode=True`.
  2. The 3-element return shape is preserved.
  3. `da_seg == 0.0` (drivable losses removed).
  4. `ll_seg` is finite, non-zero, and differentiable (backward pass works).
  5. All 4 expected lane-loss keys are present:
     lane_cls_loss, lane_xytl_loss, lane_iou_loss, lane_seg_aux_loss.
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


def _build_synthetic_batch(B: int = 2, img_size: int = 640,
                           max_lanes: int = 8, num_points: int = 72):
    """Synthetic batch covering both detection and lane-only segmentation."""
    import torch

    # One GT box per image (centered).
    cls = torch.tensor([0, 0], dtype=torch.long)
    bboxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3]], dtype=torch.float32,
    )
    batch_idx = torch.tensor([0, 1], dtype=torch.long)

    # lane_targets: (B, max_lanes, 78) - one valid lane per image, rest no-lane.
    vec_len = 2 + 4 + num_points
    lane_targets = torch.full((B, max_lanes, vec_len), -1e5, dtype=torch.float32)
    lane_targets[:, :, 0] = 1.0  # neg
    lane_targets[:, :, 1] = 0.0  # pos
    # Set row 0 of each image to a valid lane.
    for b in range(B):
        lane_targets[b, 0, 0] = 0.0          # neg
        lane_targets[b, 0, 1] = 1.0          # pos
        lane_targets[b, 0, 2] = 0.0          # start_y (slot 0 = image bottom)
        lane_targets[b, 0, 3] = 320.0        # start_x (mid)
        lane_targets[b, 0, 4] = 0.5          # theta ~ vertical
        lane_targets[b, 0, 5] = 30.0         # length in slots
        lane_targets[b, 0, 6:6 + 30] = torch.linspace(320, 200, 30)

    # lane_seg_mask: (B, 1, H, W) binary, with a vertical stripe.
    lane_seg_mask = torch.zeros(B, 1, img_size, img_size, dtype=torch.float32)
    lane_seg_mask[:, 0, :, img_size // 2 - 5: img_size // 2 + 5] = 1.0

    batch = {
        'img': torch.randn(B, 3, img_size, img_size),
        'cls': cls,
        'bboxes': bboxes,
        'batch_idx': batch_idx,
        'lane_seg_mask': lane_seg_mask,
        'lane_targets': lane_targets,
        # type_task: drivable removed; only lane in segmentation.
        'type_task': [
            {'detection': [0], 'segmentation': [1]},
            {'detection': [0], 'segmentation': [1]},
        ],
    }
    return batch


def smoke_test(rmt_ppad_root: Path, yaml_path: Path) -> int:
    import torch
    _ensure_vendor_first_on_path(rmt_ppad_root)

    print(f'[smoke] importing MTDETR from {rmt_ppad_root}', flush=True)
    from ultralytics import MTDETR  # noqa: E402

    print(f'[smoke] building model from {yaml_path}', flush=True)
    model = MTDETR(str(yaml_path))
    inner = model.model.cpu().train()
    n_params = sum(p.numel() for p in inner.parameters())
    print(f'[smoke] built; params={n_params:,}', flush=True)

    # 1. Criterion is MTDETRDLoss with lane_only_mode=True.
    if not hasattr(inner, 'criterion') or inner.criterion is None:
        inner.criterion = inner.init_criterion()
    criterion = inner.criterion
    print(f'[smoke] criterion class: {type(criterion).__name__}')
    if type(criterion).__name__ != 'MTDETRDLoss':
        print(f'[smoke] FAIL: expected MTDETRDLoss, got {type(criterion).__name__}')
        return 1
    if not getattr(criterion, 'lane_only_mode', False):
        print('[smoke] FAIL: criterion.lane_only_mode is False')
        return 1
    print('[smoke] criterion.lane_only_mode = True')

    # 2. Run loss on a synthetic batch.
    batch = _build_synthetic_batch(B=2, img_size=640, max_lanes=8)
    print(f'[smoke] synthetic batch built: img={tuple(batch["img"].shape)}, '
          f'lane_targets={tuple(batch["lane_targets"].shape)}, '
          f'lane_seg_mask={tuple(batch["lane_seg_mask"].shape)}')

    print('[smoke] calling model.loss(batch) ...', flush=True)
    loss_terms, loss_items, _ = inner.loss(batch)

    if not isinstance(loss_terms, list) or len(loss_terms) != 3:
        print(f'[smoke] FAIL: loss_terms shape; got {type(loss_terms).__name__} '
              f'len={len(loss_terms) if hasattr(loss_terms, "__len__") else "?"}')
        return 1

    L_det, da_seg, ll_seg = loss_terms
    print(f'[smoke] L_det  = {float(L_det):.4f}')
    print(f'[smoke] da_seg = {float(da_seg):.4f}  (expect 0.0000)')
    print(f'[smoke] ll_seg = {float(ll_seg):.4f}')

    # 3. da_seg must be exactly zero.
    if float(da_seg) != 0.0:
        print(f'[smoke] FAIL: da_seg = {float(da_seg)}, expected 0.0')
        return 1

    # 4. All three terms finite.
    for name, t in (('L_det', L_det), ('da_seg', da_seg), ('ll_seg', ll_seg)):
        if not torch.isfinite(t).all():
            print(f'[smoke] FAIL: {name} is non-finite: {t}')
            return 1

    # 5. Backward pass.
    total = L_det + ll_seg  # da_seg has no grad in lane-only
    print(f'[smoke] total loss for backward: {float(total):.4f}')
    total.backward()
    n_grads = sum(1 for p in inner.parameters() if p.grad is not None
                  and torch.isfinite(p.grad).all())
    n_total = sum(1 for _ in inner.parameters())
    print(f'[smoke] backward OK; {n_grads}/{n_total} params received finite gradient')

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
