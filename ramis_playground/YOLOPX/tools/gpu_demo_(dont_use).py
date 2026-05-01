# tools/fast_demo.py
import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from numpy import random
from tqdm import tqdm

from lib.config import cfg
from lib.utils.utils import select_device
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
from lib.core.postprocess import morphological_process, connect_lane

# ---- same transform as demo.py ----
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def process(cfg, opt):
    device = select_device(None, opt.device)
    half = device.type != 'cpu'
    os.makedirs(opt.save_dir, exist_ok=True)

    # --- Load model (identical to demo.py) ---
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()

    # Warmup
    dummy = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(dummy.half() if half else dummy)
    model.eval()

    names  = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    dataset  = LoadImages(opt.source, img_size=opt.img_size)
    vid_path, vid_writer = None, None

    # --- Batch buffer ---
    batch_imgs     = []   # preprocessed tensors  (C,H,W)
    batch_img_dets = []   # original frames for drawing
    batch_shapes   = []   # padding/ratio info
    batch_paths    = []
    batch_vid_caps = []

    def run_batch(batch_imgs, batch_img_dets, batch_shapes, batch_paths, batch_vid_caps):
        nonlocal vid_path, vid_writer

        imgs_tensor = torch.stack(batch_imgs).to(device)
        imgs_tensor = imgs_tensor.half() if half else imgs_tensor.float()

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(imgs_tensor)
            inf_out, _ = det_out
            det_preds = non_max_suppression(
                inf_out,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                classes=None,
                agnostic=False
            )

        _, _, height, width = imgs_tensor.shape

        for idx in range(len(batch_imgs)):
            path    = batch_paths[idx]
            img_det = batch_img_dets[idx].copy()
            shapes  = batch_shapes[idx]
            vid_cap = batch_vid_caps[idx]
            det     = det_preds[idx]

            h, w, _ = img_det.shape
            pad_w, pad_h = shapes[1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)

            # --- Drivable area mask ---
            da_predict  = da_seg_out[idx:idx+1, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(da_predict, size=(h, w), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze()

            # --- Lane mask ---
            ll_predict  = ll_seg_out[idx:idx+1, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(ll_predict, size=(h, w), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze()

            # Remove lane overlap from drivable area
            da_seg_mask = da_seg_mask - ll_seg_mask
            road = torch.zeros_like(da_seg_mask)
            road[da_seg_mask == 1] = 1
            da_seg_mask = road.cpu().numpy()
            ll_seg_mask = ll_seg_mask.cpu().numpy()

            # --- Draw segmentation ---
            img_det = img_det[..., ::-1]  # BGR→RGB
            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

            # --- Draw detections ---
            if len(det):
                det[:, :4] = scale_coords(imgs_tensor.shape[2:], det[:, :4], img_det.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{conf:.2f}'
                    plot_one_box(xyxy, img_det, label=label, color=(0, 255, 255), line_thickness=2)

            # --- Save output ---
            save_path = str(Path(opt.save_dir) / Path(path).name)
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fps_out = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                h_out, w_out, _ = img_det.shape
                vid_writer = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps_out,
                    (w_out, h_out)
                )
            vid_writer.write(img_det)

    # --- Main loop ---
    t_start = time.time()
    total   = len(dataset)

    for path, img, img_det, vid_cap, shapes in tqdm(dataset, total=total):
        img_tensor = transform(img)
        batch_imgs.append(img_tensor)
        batch_img_dets.append(img_det)
        batch_shapes.append(shapes)
        batch_paths.append(path)
        batch_vid_caps.append(vid_cap)

        if len(batch_imgs) == opt.batch_size:
            run_batch(batch_imgs, batch_img_dets, batch_shapes, batch_paths, batch_vid_caps)
            batch_imgs.clear()
            batch_img_dets.clear()
            batch_shapes.clear()
            batch_paths.clear()
            batch_vid_caps.clear()

    # Flush remaining frames
    if batch_imgs:
        run_batch(batch_imgs, batch_img_dets, batch_shapes, batch_paths, batch_vid_caps)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    elapsed = time.time() - t_start
    print(f"\nDone — {total} frames in {elapsed:.1f}s ({total/elapsed:.1f} fps)")
    print(f"Output saved to: {opt.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str,   default='weights/epoch-195.pth')
    parser.add_argument('--source',     type=str,   default='inputs/dashcam_freeway.mp4')
    parser.add_argument('--save-dir',   type=str,   default='outputs/')
    parser.add_argument('--img-size',   type=int,   default=640)
    parser.add_argument('--batch-size', type=int,   default=256)
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres',  type=float, default=0.45)
    parser.add_argument('--device',     type=str,   default='0')
    opt = parser.parse_args()
    process(cfg, opt)
