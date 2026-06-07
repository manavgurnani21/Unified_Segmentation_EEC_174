"""Phase P4 acceptance test.

Builds the lane-only MTDETR model from `rtdetr-l_bdd_clr_lane.yaml`, runs a
forward pass on a dummy (1, 3, 640, 640) tensor, and asserts:

  1. The build succeeds.
  2. `model.seg_head` is a `LaneSegHead`.
  3. `model.seg_head.adapter` is a `GCAtoCLRAdapter`.
  4. `model.seg_head.lane_head` is a `CLRHeadForSquareImage`.
  5. NO `drivable` named-module anywhere in the model tree.
  6. Forward (eval mode) returns without crashing and the seg output
     is the dict shape produced by LaneSegHead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_vendor_first_on_path(rmt_ppad_root: Path) -> None:
    """Insert vendor/RMT-PPAD/ at sys.path[0] so the modified `ultralytics`
    package is imported instead of any global pip install."""
    p = str(rmt_ppad_root.resolve())
    # Drop any prior ultralytics entries.
    sys.path = [x for x in sys.path if 'ultralytics' not in x.lower()]
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    # Drop any cached ultralytics modules.
    for k in list(sys.modules):
        if k == 'ultralytics' or k.startswith('ultralytics.'):
            del sys.modules[k]


def smoke_test(rmt_ppad_root: Path, yaml_path: Path) -> int:
    import torch

    _ensure_vendor_first_on_path(rmt_ppad_root)

    print(f'[smoke] importing MTDETR from {rmt_ppad_root}', flush=True)
    from ultralytics import MTDETR  # noqa: E402

    print(f'[smoke] building model from {yaml_path}', flush=True)
    model = MTDETR(str(yaml_path))
    inner = model.model.cpu().eval()
    n_params = sum(p.numel() for p in inner.parameters())
    print(f'[smoke] built; total params = {n_params:,}', flush=True)

    # 1. Find seg_head.
    seg_head = None
    seg_head_name = None
    for name, m in inner.named_modules():
        # The detector module is at the very end; its seg_head attribute holds ours.
        if name.endswith('.seg_head') and seg_head is None:
            seg_head = m
            seg_head_name = name
    if seg_head is None:
        print('[smoke] FAIL: no .seg_head found in named_modules')
        return 1
    print(f'[smoke] seg_head at "{seg_head_name}": {type(seg_head).__name__}')

    expected = 'LaneSegHead'
    if type(seg_head).__name__ != expected:
        print(f'[smoke] FAIL: seg_head is {type(seg_head).__name__}, expected {expected}')
        return 1

    # 2. Confirm adapter + lane_head children.
    children = dict(seg_head.named_children())
    if 'adapter' not in children:
        print(f'[smoke] FAIL: seg_head.adapter missing; children = {list(children)}')
        return 1
    if 'lane_head' not in children:
        print(f'[smoke] FAIL: seg_head.lane_head missing; children = {list(children)}')
        return 1
    print(f'[smoke] seg_head.adapter:   {type(children["adapter"]).__name__}')
    print(f'[smoke] seg_head.lane_head: {type(children["lane_head"]).__name__}')

    if type(children['adapter']).__name__ != 'GCAtoCLRAdapter':
        print(f'[smoke] FAIL: adapter is {type(children["adapter"]).__name__}')
        return 1
    if type(children['lane_head']).__name__ != 'CLRHeadForSquareImage':
        print(f'[smoke] FAIL: lane_head is {type(children["lane_head"]).__name__}')
        return 1

    # 3. No `drivable` modules anywhere.
    bad = [name for name, _ in inner.named_modules() if 'drivable' in name.lower()]
    if bad:
        print(f'[smoke] FAIL: drivable modules still present: {bad[:10]}')
        return 1
    print('[smoke] no `drivable` modules anywhere in named_modules')

    # 4. Eval forward.
    print('[smoke] running eval forward on (1, 3, 640, 640) ...', flush=True)
    fake_img = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = inner(fake_img)
    print(f'[smoke] eval forward returned; outer type = {type(out).__name__}')

    # In eval, MTDETRDecoder returns y or (y, x) depending on export flag.
    # x is a tuple where seg_mask is one of the elements.
    if isinstance(out, tuple) and len(out) == 2:
        y_pred, x_tuple = out
        print(f'[smoke] y_pred shape: {tuple(y_pred.shape) if hasattr(y_pred, "shape") else "n/a"}')
        # x_tuple layout from MTDETRDecoder.forward:
        #   (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, seg_mask, gate_mean)
        if len(x_tuple) >= 6:
            seg_mask = x_tuple[5]
            print(f'[smoke] seg_mask type: {type(seg_mask).__name__}')
            if not isinstance(seg_mask, dict):
                print(f'[smoke] FAIL: expected seg_mask to be dict (LaneSegHead output), '
                      f'got {type(seg_mask).__name__}')
                return 1
            if 'lane_output' not in seg_mask:
                print(f'[smoke] FAIL: seg_mask missing "lane_output"; keys = {list(seg_mask)}')
                return 1
            lane_pred = seg_mask['lane_output']
            print(f'[smoke] lane_output shape: '
                  f'{tuple(lane_pred.shape) if hasattr(lane_pred, "shape") else "list"}')

    print('\n[smoke] PASS')
    return 0


def _default_paths():
    here = Path(__file__).resolve()
    # tools/ -> P4_lane_head_integration/ -> rmt_ppad_migration/ -> stage2/ -> yolop_vehicle_lane/
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
