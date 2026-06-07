"""Phase P6 acceptance test.

Builds a YOLODataset over a minimal on-disk slice (10 hardlinked JPGs + 10
corresponding .pt files) and asserts the collated batch carries:
  - `img`           shape (B, 3, H, W)
  - `lane_seg_mask` shape (B, 1, H, W) - rasterized from `lane_targets`
  - `lane_targets`  shape (B, max_lanes, 78)
  - NO `merge_mask[:,0]` being consumed (drivable channel)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
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


def _make_tiny_dataset(out_root: Path, n_samples: int = 10, imgsz: int = 640):
    """Create a tiny on-disk dataset directory:
        out_root/images/train/000000.jpg ...
        out_root/labels/train/000000.txt (empty - no bbox needed)
        out_root/lane_targets/train/000000.pt
    """
    import cv2
    import numpy as np
    import torch

    img_dir = out_root / 'images' / 'train'
    lbl_dir = out_root / 'labels' / 'train'
    lt_dir = out_root / 'lane_targets' / 'train'
    for d in (img_dir, lbl_dir, lt_dir):
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    for i in range(n_samples):
        stem = f'{i:06d}'
        # Random RGB image.
        img = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f'{stem}.jpg'), img)
        # Empty label (just a single detection class 0 at center to avoid empty batch path).
        with open(lbl_dir / f'{stem}.txt', 'w', encoding='utf-8') as fh:
            fh.write('0 0.5 0.5 0.2 0.2\n')
        # Synthetic lane target: row 0 has a near-vertical lane.
        target = torch.full((8, 78), -1e5, dtype=torch.float32)
        target[:, 0] = 1.0
        target[:, 1] = 0.0
        target[0, 0] = 0.0
        target[0, 1] = 1.0
        target[0, 2] = 0.0
        target[0, 3] = 320.0
        target[0, 4] = 0.5
        target[0, 5] = 30.0
        target[0, 6:6 + 30] = torch.linspace(320, 360, 30)
        torch.save(target, lt_dir / f'{stem}.pt')


def smoke_test(rmt_ppad_root: Path) -> int:
    import torch
    _ensure_vendor_first_on_path(rmt_ppad_root)

    # P6: use MTDETRDataset, not YOLODataset. The base YOLODataset's val
    # build_transforms() chains LetterBox -> Format, but LetterBox dies on
    # the [img, seg_mask] list that update_labels_info packs. MTDETRDataset
    # overrides build_transforms() to use Compose([]) -> Format directly
    # in eval mode (load_image already produces a correctly-sized image).
    # P0's working baseline uses this same MTDETRDataset path.
    from ultralytics.models.mtdetr.val import MTDETRDataset
    from torch.utils.data import DataLoader

    tmp_root = Path(tempfile.mkdtemp(prefix='p6_ds_'))
    try:
        print(f'[smoke] building tiny dataset at {tmp_root}', flush=True)
        _make_tiny_dataset(tmp_root, n_samples=10, imgsz=640)

        # Minimal data dict mirroring BDD_lane_only.yaml.
        data = {
            'path': str(tmp_root),
            'train': 'images/train',
            'val': 'images/train',
            'names': {0: 'vehicle', 1: 'lane'},
            'type_task': {'detection': [0], 'segmentation': [1]},
            'lane_targets_root': str(tmp_root / 'lane_targets'),
            'lane_targets_split_map': {},
            'nc': 2,
            'channels': 3,
        }

        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        hyp = get_cfg(DEFAULT_CFG)
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.degrees = 0.0
        hyp.translate = 0.0
        hyp.scale = 0.0
        hyp.shear = 0.0
        hyp.perspective = 0.0
        hyp.flipud = 0.0
        hyp.fliplr = 0.0

        print('[smoke] instantiating MTDETRDataset (task=multi)', flush=True)
        ds = MTDETRDataset(
            img_path=str(tmp_root / 'images' / 'train'),
            data=data,
            task='multi',
            imgsz=640,
            augment=False,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix='[P6 smoke]',
        )
        print(f'[smoke] dataset size = {len(ds)}', flush=True)

        # collate_fn lives on YOLODataset (parent); MTDETRDataset inherits it.
        loader = DataLoader(ds, batch_size=2, collate_fn=MTDETRDataset.collate_fn,
                            num_workers=0)
        batch = next(iter(loader))

        print(f'[smoke] batch keys: {sorted(batch.keys())}')
        for k in ['img', 'merge_mask', 'lane_seg_mask', 'lane_targets']:
            if k in batch and hasattr(batch[k], 'shape'):
                print(f'[smoke]   {k:16s} shape={tuple(batch[k].shape)} dtype={batch[k].dtype}')
            elif k in batch:
                print(f'[smoke]   {k:16s} type={type(batch[k]).__name__}')
            else:
                print(f'[smoke]   {k:16s} <missing>')

        # Acceptance assertions.
        if 'img' not in batch or tuple(batch['img'].shape) != (2, 3, 640, 640):
            print('[smoke] FAIL: img shape')
            return 1
        if 'lane_targets' not in batch:
            print('[smoke] FAIL: lane_targets missing')
            return 1
        if tuple(batch['lane_targets'].shape) != (2, 8, 78):
            print(f'[smoke] FAIL: lane_targets shape {tuple(batch["lane_targets"].shape)}')
            return 1
        if 'lane_seg_mask' not in batch:
            print('[smoke] FAIL: lane_seg_mask missing')
            return 1
        lsm = batch['lane_seg_mask']
        # merge_mask was (B, n_seg, H, W). In lane-only mode n_seg=1.
        if lsm.dim() != 4 or lsm.shape[0] != 2 or lsm.shape[1] != 1:
            print(f'[smoke] FAIL: lane_seg_mask shape {tuple(lsm.shape)}, '
                  f'expected (2, 1, H, W)')
            return 1
        # lane_seg_mask should be non-empty (we drew lanes).
        n_lane_px = int((lsm > 0).sum())
        print(f'[smoke] lane_seg_mask lane pixels = {n_lane_px}')
        if n_lane_px == 0:
            print('[smoke] FAIL: lane_seg_mask is all-zero - rasterizer failed')
            return 1

        print('\n[smoke] PASS')
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _default_paths():
    here = Path(__file__).resolve()
    rmt_ppad_root = here.parent.parent.parent / 'vendor' / 'RMT-PPAD'
    return rmt_ppad_root


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--rmt-ppad-root', type=Path, default=_default_paths())
    args = p.parse_args()
    return smoke_test(args.rmt_ppad_root)


if __name__ == '__main__':
    sys.exit(main())
