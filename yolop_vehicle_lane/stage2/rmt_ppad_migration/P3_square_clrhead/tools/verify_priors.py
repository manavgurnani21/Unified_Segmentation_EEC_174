"""Phase P3 acceptance test: render the 192 CLRHeadForSquareImage priors as
arrows on a 640x640 canvas, plus check the per-region counts and the
prior-vector tensor shape.

Outputs:
  - priors_square.png  (visual)
  - prints a per-region count summary
  - exits 0 only if 32-left + 128-bottom + 32-right and shape (192, 78)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from clr_head_square import CLRHeadForSquareImage  # noqa: E402


def _fake_cfg(img_size: int = 640, num_classes: int = 2):
    """Minimal SimpleNamespace cfg covering everything CLRHead.__init__ reads."""
    return SimpleNamespace(
        img_w=img_size,
        img_h=img_size,
        num_classes=num_classes,
        bg_weight=0.4,
        ignore_label=255,
        haskey=lambda k: False,
    )


def smoke_test(out_dir: Path, img_size: int = 640) -> int:
    print('[smoke] CLRHeadForSquareImage starting', flush=True)
    import torch
    import numpy as np

    cfg = _fake_cfg(img_size=img_size)
    head = CLRHeadForSquareImage(
        num_points=72, prior_feat_channels=64, fc_hidden_dim=64,
        num_priors=192, num_fc=2, refine_layers=3, sample_points=36, cfg=cfg,
    )
    print(f'[smoke] instantiated CLRHeadForSquareImage, '
          f'params={sum(p.numel() for p in head.parameters()):,}', flush=True)

    # Check the raw embedding table.
    emb = head.prior_embeddings.weight.detach()
    print(f'[smoke] prior_embeddings shape: {tuple(emb.shape)}  '
          f'(expect (192, 3))', flush=True)
    if tuple(emb.shape) != (192, 3):
        print('[smoke] FAIL: wrong embedding shape')
        return 1

    # Region partition: row i's start_x tells us which group.
    start_x_norm = emb[:, 1].numpy()
    n_left = int((start_x_norm == 0.0).sum())
    n_right = int((start_x_norm == 1.0).sum())
    n_bottom = int(((start_x_norm > 0.0) & (start_x_norm < 1.0)).sum())
    print(f'[smoke] per-region counts: left={n_left}  bottom={n_bottom}  right={n_right}  '
          f'(expect 32 / 128 / 32, total 192)')
    if (n_left, n_bottom, n_right) != (32, 128, 32):
        print('[smoke] FAIL: per-region partition mismatch')
        return 1

    # Generate full priors (192, 78). Validate shape.
    priors, priors_on_featmap = head.generate_priors_from_embeddings()
    print(f'[smoke] priors shape: {tuple(priors.shape)}  '
          f'(expect (192, 78))')
    if tuple(priors.shape) != (192, 78):
        print('[smoke] FAIL: wrong priors shape')
        return 1

    # Render to PNG.
    # CLRHead convention: start_y is measured from the IMAGE BOTTOM.
    #   start_y = 0  -> lane starts at image bottom  -> pixel_y = img_size
    #   start_y = 1  -> lane starts at image top     -> pixel_y = 0
    # Slope (lane goes "up" in image): from the head's own formula,
    #   x(prior_y) = start_x + (1 - prior_y - start_y) * (H / W) / tan(theta*pi)
    # so over a normalized vertical step of `dy_norm` toward the top of the image,
    # the x change is  dy_norm * H / tan(theta*pi)  (W cancels in pixel space when
    # we map x_norm -> pixel_x via * W).
    try:
        import cv2
    except ImportError:
        print('[smoke] cv2 missing - skipping PNG render')
    else:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        H = float(img_size)
        W = float(img_size)
        dy_norm = 0.3  # how far "up" the lane to draw the arrow tip (30% of image height)
        for idx in range(int(priors.shape[0])):
            sy_norm = float(priors[idx, 2])
            sx_norm = float(priors[idx, 3])
            theta_norm = float(priors[idx, 4])
            theta_rad = theta_norm * math.pi
            start_x_px = sx_norm * W
            start_y_px = (1.0 - sy_norm) * H        # FLIPPED: bottom-up
            # cotangent of theta determines lane horizontality
            tan_theta = math.tan(theta_rad) if abs(math.tan(theta_rad)) > 1e-6 else 1e-6
            dx_px = dy_norm * H / tan_theta           # signed: < 0 means lane heads left
            dy_px = -dy_norm * H                       # arrow goes UP in pixel coords
            end_x_px = start_x_px + dx_px
            end_y_px = start_y_px + dy_px
            if sx_norm == 0.0:
                color = (60, 255, 60)    # BGR green-ish    (left edge)
            elif sx_norm == 1.0:
                color = (60, 60, 255)    # BGR red-ish      (right edge)
            else:
                color = (255, 255, 60)   # BGR cyan-ish     (bottom edge)
            cv2.arrowedLine(
                img,
                (int(round(start_x_px)), int(round(start_y_px))),
                (int(round(end_x_px)), int(round(end_y_px))),
                color, 1, tipLength=0.18,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'priors_square.png'
        cv2.imwrite(str(out_path), img)
        print(f'[smoke] wrote {out_path}  '
              f'(green=left edge, cyan=bottom edge, red=right edge; arrows point into canvas)')

    print('[smoke] PASS')
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='.',
                   help='Where to write priors_square.png')
    p.add_argument('--img_size', type=int, default=640)
    args = p.parse_args()
    return smoke_test(Path(args.out_dir), img_size=args.img_size)


if __name__ == '__main__':
    sys.exit(main())
