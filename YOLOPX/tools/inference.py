import argparse
import os
import sys
from pathlib import Path
import threading
import queue as Queue

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import ffmpeg
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from numpy import random
from tqdm import tqdm

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from lib.config import cfg, update_config
from lib.core.general import non_max_suppression, scale_coords
from lib.dataset import LoadImages, LoadStreams
from lib.models import get_net
from lib.utils import plot_one_box, show_seg_result
from lib.utils.utils import select_device

# ── ImageNet normalization constants ──────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path, device):
    model = get_net(cfg)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if device.type != 'cpu':
        model.half()
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GPU letterbox  (replaces CPU letterbox inside LoadImages)
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_gpu(frame_hwc: torch.Tensor, target: int = 640):
    """
    Letterbox a [H, W, C] uint8 GPU tensor to [1, 3, target, target] float16/32.
    Returns (tensor, ratio, (pad_w, pad_h)) — shapes format matches LoadImages.
    """
    h, w = frame_hwc.shape[:2]
    r = min(target / h, target / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    pad_h = (target - new_h) / 2
    pad_w = (target - new_w) / 2
    top    = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left   = int(round(pad_w - 0.1))
    right  = int(round(pad_w + 0.1))

    # [H, W, C] uint8 → [1, C, H, W] float
    x = frame_hwc.float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x = F.pad(x, (left, right, top, bottom), value=114 / 255.0)

    # Normalize in-place
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    x = (x - mean) / std

    return x, r, (pad_w, pad_h)


# ─────────────────────────────────────────────────────────────────────────────
# DALI video decode pipeline  (nvdec → GPU tensor)
# ─────────────────────────────────────────────────────────────────────────────

@pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=4)
def _dali_video_pipeline(video_path):
    frames = fn.readers.video(
        filenames=[video_path],
        sequence_length=1,
        stride=1,
        device="gpu",
        random_shuffle=False,
        file_list_include_preceding_frame=False,
        name="VideoReader"
    )
    return frames


def build_video_loader(video_path, gpu_id=0):
    pipe = _dali_video_pipeline(video_path, device_id=gpu_id)
    pipe.build()
    return DALIGenericIterator(
        pipe,
        output_map=["frames"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=False,
        reader_name="VideoReader"
    )


# ─────────────────────────────────────────────────────────────────────────────
# nvenc encoder  (replaces cv2.VideoWriter)
# ─────────────────────────────────────────────────────────────────────────────

def build_nvenc_writer(save_path, width, height, fps):
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24',
               s=f'{width}x{height}', r=fps)
        .output(save_path, vcodec='h264_nvenc', pix_fmt='yuv420p',
                preset='p4', cq=23)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared postprocessing  (unchanged logic from original demo.py)
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(img_tensor, img_det, det_out, da_seg_out, ll_seg_out,
                shapes, opt):
    """
    img_det  : BGR numpy [H, W, 3] — original-resolution frame for drawing
    shapes   : ((orig_h, orig_w), ((ratio, ratio), (pad_w, pad_h)))
    Returns annotated BGR numpy frame.
    """
    inf_out, _ = det_out
    det_pred = non_max_suppression(
        inf_out, conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres, classes=None, agnostic=False
    )
    det = det_pred[0]

    _, _, height, width = img_tensor.shape
    h, w, _  = img_det.shape
    pad_w    = int(shapes[1][1][0])
    pad_h    = int(shapes[1][1][1])

    # Drivable-area mask
    da_predict  = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, size=(h, w), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze()

    # Lane-line mask
    ll_predict  = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, size=(h, w), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze()

    # Subtract lane from drivable area to avoid overlap
    road = torch.zeros_like(da_seg_mask)
    road[da_seg_mask - ll_seg_mask == 1] = 1
    da_seg_mask = road.cpu().numpy()
    ll_seg_mask = ll_seg_mask.cpu().numpy()

    # BGR → RGB for show_seg_result, then draw detections
    img_det = img_det[..., ::-1].copy()
    img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), 0, 0, is_demo=True)

    if len(det):
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img_det.shape).round()
        for *xyxy, conf, cls in reversed(det):
            plot_one_box(xyxy, img_det, label=f'{conf:.2f}',
                         color=(0, 255, 255), line_thickness=2)

    # RGB → BGR for cv2 / ffmpeg
    return img_det[..., ::-1].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Threading stuff
# ─────────────────────────────────────────────────────────────────────────────

def _visualization_worker(q, enc):
    while True:
        item = q.get()
        if item is None:
            break

        vis_frame, det, da_seg_mask, ll_seg_mask, img_tensor_shape = item

        img_det = vis_frame[..., ::-1].copy()  # RGB → BGR
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask),
                                  0, 0, is_demo=True)
        if len(det):
            det[:, :4] = scale_coords(img_tensor_shape,
                                      det[:, :4], img_det.shape).round()
            for *xyxy, conf, cls in reversed(det):
                plot_one_box(xyxy, img_det, label=f'{conf:.2f}',
                             color=(0, 255, 255), line_thickness=2)

        enc.stdin.write(img_det[..., ::-1].tobytes())  # BGR → RGB → ffmpeg
        q.task_done()
# ─────────────────────────────────────────────────────────────────────────────
# Per-source detect paths
# ─────────────────────────────────────────────────────────────────────────────

def detect_video(opt, model, device):
    gpu_id = int(opt.device) if opt.device.isnumeric() else 0
    loader = build_video_loader(opt.source, gpu_id=gpu_id)

    cap = cv2.VideoCapture(opt.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    save_path = str(Path(opt.save_dir) / Path(opt.source).name)
    enc  = build_nvenc_writer(save_path, opt.img_size, opt.img_size, fps)
    half = device.type != 'cpu'

    vis_queue  = Queue.Queue(maxsize=8)
    vis_thread = threading.Thread(
        target=_visualization_worker,
        args=(vis_queue, enc),
        daemon=True
    )
    vis_thread.start()

    for batch in tqdm(loader, desc='video'):
        frame_gpu = batch[0]['frames'][0][0]  # [H, W, C] uint8 RGB on GPU

        # Letterbox + normalize on GPU
        img_tensor, ratio, (pad_w, pad_h) = letterbox_gpu(frame_gpu, opt.img_size)
        if half:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img_tensor)

        # Masks at 640x640 — no full-res interpolation
        da_seg_mask = F.interpolate(da_seg_out, size=(opt.img_size, opt.img_size),
                                    mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze()

        ll_seg_mask = F.interpolate(ll_seg_out, size=(opt.img_size, opt.img_size),
                                    mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze()

        road = torch.zeros_like(da_seg_mask)
        road[da_seg_mask - ll_seg_mask == 1] = 1

        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres,
                                       iou_thres=opt.iou_thres,
                                       classes=None, agnostic=False)

        # Denormalize img_tensor → uint8 numpy for visualization (stays 640x640)
        vis_frame = img_tensor[0].float()
        vis_frame = (vis_frame * _STD.to(device).view(3,1,1)
                     + _MEAN.to(device).view(3,1,1))
        vis_frame = vis_frame.clamp(0, 1).mul(255).byte()
        vis_frame = vis_frame.permute(1, 2, 0).cpu().numpy()  # [640, 640, 3] RGB

        vis_queue.put((
            vis_frame,
            det_pred[0],
            road.cpu().numpy(),
            ll_seg_mask.cpu().numpy(),
            img_tensor.shape[2:]
        ))

    vis_queue.join()
    vis_queue.put(None)
    vis_thread.join()

    enc.stdin.close()
    enc.wait()
    print(f'Video saved → {save_path}')


def detect_images(opt, model, device):
    """Standard image-folder inference (DALI not warranted for static images)."""
    dataset  = LoadImages(opt.source, img_size=opt.img_size)
    half     = device.type != 'cpu'
    mean     = _MEAN.to(device)
    std      = _STD.to(device)

    for path, img, img_det, _, shapes in tqdm(dataset, total=len(dataset), desc='images'):
        # img is HWC uint8 numpy from LoadImages (already letterboxed)
        x = torch.from_numpy(img).float().div_(255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x - mean) / std
        if half:
            x = x.half()

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(x)

        result = postprocess(x, img_det, det_out, da_seg_out, ll_seg_out, shapes, opt)
        save_path = str(Path(opt.save_dir) / Path(path).name)
        cv2.imwrite(save_path, result)

    print(f'Images saved → {opt.save_dir}')


def detect_stream(opt, model, device):
    """Webcam / RTSP stream inference (DALI cannot consume live streams)."""
    cudnn.benchmark = True
    dataset  = LoadStreams(opt.source, img_size=opt.img_size)
    half     = device.type != 'cpu'
    mean     = _MEAN.to(device)
    std      = _STD.to(device)

    for _, img, img_det, _, shapes in dataset:
        x = torch.from_numpy(img).float().div_(255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x - mean) / std
        if half:
            x = x.half()

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(x)

        result = postprocess(x, img_det[0], det_out, da_seg_out, ll_seg_out, shapes, opt)
        cv2.imshow('stream', result)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect(opt):
    device = select_device(None, opt.device)
    os.makedirs(opt.save_dir, exist_ok=True)

    model = load_model(opt.weights, device)

    # Warmup run
    dummy = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    if device.type != 'cpu':
        dummy = dummy.half()
    with torch.no_grad():
        model(dummy)

    source = opt.source
    if str(source).isnumeric() or source.startswith('rtsp'):
        detect_stream(opt, model, device)
    elif Path(source).suffix.lower() in VIDEO_EXTS:
        detect_video(opt, model, device)
    else:
        detect_images(opt, model, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOPX inference with DALI + nvenc')
    parser.add_argument('--weights',    type=str,   required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--source',     type=str,   required=True,
                        help='Input: video file, image folder, webcam index, or rtsp://')
    parser.add_argument('--save-dir',   type=str,   default='inference/output',
                        help='Directory to write results')
    parser.add_argument('--img-size',   type=int,   default=640,
                        help='Inference resolution (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--iou-thres',  type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--device',     type=str,   default='0',
                        help='CUDA device id (e.g. 0) or "cpu"')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)