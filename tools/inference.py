# python3 tools/inference.py --weights weights/epoch-195.pth --source inputs/dashcam_freeway.mp4 --save-dir inference/video_output --batch-size 16 --device 0

import argparse, os, sys, time, threading
from pathlib import Path
from queue import Queue

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from numpy import random

from lib.config import cfg
from lib.utils.utils import select_device
from lib.models import get_net
from lib.dataset import LoadImages
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box

SENTINEL = None


def gpu_seg_overlay(frames_gpu, da_masks, ll_masks):
    """
    Replaces show_seg_result — runs on GPU across the whole batch at once.
    frames_gpu : (N, H, W, 3) float32 on GPU, BGR, values 0-255
    da_masks   : (N, H, W)    uint8  on GPU
    ll_masks   : (N, H, W)    uint8  on GPU
    returns    : (N, H, W, 3) uint8  on GPU, BGR
    """
    # Color overlay: green=[0,255,0] BGR for drivable, blue=[255,0,0] BGR for lanes
    green = torch.tensor([0, 255, 0],   device=frames_gpu.device, dtype=torch.float32)
    blue  = torch.tensor([255, 0, 0],   device=frames_gpu.device, dtype=torch.float32)

    da_3d = (da_masks == 1).unsqueeze(-1)              # (N,H,W,1) bool
    ll_3d = (ll_masks == 1).unsqueeze(-1)              # (N,H,W,1) bool

    color = torch.zeros_like(frames_gpu)
    # DA Segmentation:
    # color = torch.where(da_3d, green.view(1,1,1,3).expand_as(color), color)
    color = torch.where(ll_3d, blue.view(1,1,1,3).expand_as(color),  color)

    # Blend only where there is color (mask != 0)
    has_color = color.mean(dim=-1, keepdim=True) != 0  # (N,H,W,1)
    blended   = torch.where(has_color,
                            frames_gpu * 0.5 + color * 0.5,
                            frames_gpu)
    return blended.byte()   # (N,H,W,3) uint8


def process(cfg, opt):
    device = select_device(None, opt.device)
    half   = device.type != 'cpu'
    os.makedirs(opt.save_dir, exist_ok=True)

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1).half()
    STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1).half()

    # --- Load model ---
    print("=== Loading model ===")
    model = get_net(cfg)
    ckpt  = torch.load(opt.weights, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    if half:
        model.half()
    dummy = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(dummy.half() if half else dummy)
    model.eval()

    dataset = LoadImages(opt.source, img_size=opt.img_size)
    total   = len(dataset)
    print(f"  {total} frames to process\n")

    decode_queue = Queue(maxsize=8)
    save_queue   = Queue(maxsize=8)

    # ------------------------------------------------------------------ #
    # Thread 1: decode + letterbox                                        #
    # ------------------------------------------------------------------ #
    def decode_thread():
        buf_imgs, buf_meta = [], []

        for path, input_img, img0, vid_cap, shapes in dataset:
            t = torch.from_numpy(
                    np.ascontiguousarray(input_img)
                ).permute(2,0,1).float() / 255.0   # (3,H,W)

            buf_imgs.append(t)
            out_h, out_w = input_img.shape[:2]   # letterboxed dims, e.g. 640x384 for 16:9
            img0_small   = cv2.resize(img0, (out_w, out_h))
            buf_meta.append((path, img0_small, shapes, vid_cap))

            if len(buf_imgs) == opt.batch_size:
                decode_queue.put((torch.stack(buf_imgs), buf_meta.copy()))
                buf_imgs.clear()
                buf_meta.clear()

        if buf_imgs:
            decode_queue.put((torch.stack(buf_imgs), buf_meta.copy()))

        decode_queue.put(SENTINEL)
        print("\n[decode] done")

    # ------------------------------------------------------------------ #
    # Thread 2 (main): inference + GPU mask postprocess + GPU blending    #
    # ------------------------------------------------------------------ #
    def inference_loop():
        frames_done = 0
        t_start     = time.time()

        while True:
            item = decode_queue.get()
            if item is SENTINEL:
                save_queue.put(SENTINEL)
                break

            batch_cpu, meta = item
            N = len(meta)

            # Upload + normalize
            batch_gpu = batch_cpu.to(device).half()
            batch_gpu = (batch_gpu - MEAN) / STD
            H, W      = batch_gpu.shape[2], batch_gpu.shape[3]

            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = model(batch_gpu)
                inf_out, _ = det_out
                det_preds  = non_max_suppression(
                    inf_out, conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres, classes=None, agnostic=False
                )

                # --- Batch mask postprocessing on GPU ---
                # Collect original frame sizes for interpolation
                orig_sizes = [(m[1].shape[0], m[1].shape[1]) for m in meta]
                pads       = [(int(m[2][1][1][0]), int(m[2][1][1][1])) for m in meta]

                # All frames in a video are the same size — use first
                pad_w, pad_h = pads[0]
                out_h, out_w = meta[0][1].shape[:2]   # read from img0_small in meta
                out_size     = (out_h, out_w)

                da_pred = da_seg_out[:, :, pad_h:(H-pad_h), pad_w:(W-pad_w)]
                ll_pred = ll_seg_out[:, :, pad_h:(H-pad_h), pad_w:(W-pad_w)]
    
                da_up = F.interpolate(da_pred.float(), size=out_size, mode='bilinear')
                ll_up = F.interpolate(ll_pred.float(), size=out_size, mode='bilinear')

                _, da_masks = torch.max(da_up, 1)   # (N,H,W)
                _, ll_masks = torch.max(ll_up, 1)

                da_masks = da_masks.int()
                ll_masks = ll_masks.int()

                # Remove lane overlap from drivable area
                da_masks = (da_masks - ll_masks).clamp(min=0)

                # --- Batch seg overlay on GPU ---
                # Stack original frames (N,H,W,3) float32 on GPU
                img0s_np  = np.stack([m[1] for m in meta])              # (N,H,W,3) uint8 BGR (already BGR from RGB→BGR flip)
                # img0 from LoadImages is RGB — convert to BGR for OpenCV output
                img0s_np  = img0s_np[..., ::-1].copy()                  # RGB→BGR
                frames_gpu = torch.from_numpy(img0s_np).to(device).float()  # (N,H,W,3)

                blended = gpu_seg_overlay(frames_gpu, da_masks, ll_masks)  # (N,H,W,3) uint8 GPU
                blended_cpu = blended.cpu().numpy()                        # back to CPU for drawing

            # Save thread only receives blended frames + detections
            save_queue.put((det_preds, blended_cpu, meta, (H, W)))

            frames_done += len(meta)
            elapsed      = time.time() - t_start
            print(f"\r[inference] {frames_done}/{total}  "
                  f"{frames_done/elapsed:.1f} fps", end='', flush=True)

        elapsed = time.time() - t_start
        print(f"\n[inference] done — {frames_done} frames in "
              f"{elapsed:.2f}s  ({frames_done/elapsed:.1f} fps)")

    # ------------------------------------------------------------------ #
    # Thread 3: draw boxes + write video (minimal CPU work)               #
    # ------------------------------------------------------------------ #
    def save_thread():
        vid_path, vid_writer = None, None
        frames_done = 0
        t_start     = time.time()

        while True:
            item = save_queue.get()
            if item is SENTINEL:
                break

            det_preds, blended_cpu, meta, (H, W) = item

            for i, (path, img0, shapes, vid_cap) in enumerate(meta):
                img_det = blended_cpu[i]   # already blended, BGR uint8
                det     = det_preds[i]

                # Draw bounding boxes (few per frame — fast)
                if len(det):
                    det[:, :4] = scale_coords(
                        (H, W), det[:, :4], img_det.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, img_det, label=f'{conf:.2f}',
                                     color=(0,255,255), line_thickness=2)

                # Write frame
                save_path = str(Path(opt.save_dir) / Path(path).name)
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    fps_out    = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                    h_o, w_o   = img_det.shape[:2]
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                        fps_out, (w_o, h_o))
                vid_writer.write(img_det)
                frames_done += 1

        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()

        elapsed = time.time() - t_start
        print(f"[save] done — {frames_done} frames in "
              f"{elapsed:.2f}s  ({frames_done/elapsed:.1f} fps)")
        print(f"Output: {opt.save_dir}")

    # --- Launch ---
    t_total = time.time()
    t1 = threading.Thread(target=decode_thread, daemon=True)
    t3 = threading.Thread(target=save_thread,   daemon=True)
    t1.start()
    t3.start()
    inference_loop()
    t1.join()
    t3.join()
    print(f"\nWall time: {time.time()-t_total:.1f}s  "
          f"({total/(time.time()-t_total):.1f} fps end-to-end)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str,   default='weights/epoch-195.pth')
    parser.add_argument('--source',     type=str,   default='inputs/')
    parser.add_argument('--save-dir',   type=str,   default='outputs/')
    parser.add_argument('--img-size',   type=int,   default=640)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres',  type=float, default=0.45)
    parser.add_argument('--device',     type=str,   default='0')
    opt = parser.parse_args()
    process(cfg, opt)
