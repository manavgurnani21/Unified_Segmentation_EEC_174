# python tools/inference_trt.py --engine weights/yolopx_fp16.trt --source inputs/video.mp4 --save-dir outputs/ --log speed_log.csv

import argparse, os, sys, time, threading, csv
from pathlib import Path
from queue import Queue

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy import random

from lib.config import cfg
from lib.dataset import LoadImages
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box

SENTINEL = None


class TRTEngine:
    def __init__(self, engine_path, device, max_batch=16):
        import tensorrt as trt
        self._trt = trt

        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device  = device

        # Identify input / output names and their expected dtypes
        self.input_name  = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        in_trt_dtype = self.engine.get_tensor_dtype(self.input_name)
        self.input_dtype = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16,
        }.get(in_trt_dtype, torch.float32)
        print(f"TRT input  '{self.input_name}': dtype={in_trt_dtype} → torch.{self.input_dtype}")

        # Dedicated CUDA stream — keeps TRT isolated from PyTorch's internal streams
        self._stream = torch.cuda.Stream(device=device)

        # Map TRT dtypes to torch dtypes
        self._dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8:  torch.int8,
            trt.DataType.BOOL:  torch.bool,
        }

        # Pre-allocate output buffers at max_batch — dtype matched to TRT output
        with torch.cuda.stream(self._stream):
            self.context.set_input_shape(self.input_name, (max_batch, 3, 640, 640))
            self.context.infer_shapes()
            self._out_bufs = []
            for name in self.output_names:
                shape     = tuple(self.context.get_tensor_shape(name))
                trt_dtype = self.engine.get_tensor_dtype(name)
                buf = torch.empty(shape,
                                  dtype=self._dtype_map.get(trt_dtype, torch.float32),
                                  device=device)
                self._out_bufs.append(buf)
                print(f"  output '{name}': shape={shape}  dtype={trt_dtype}")
        self._stream.synchronize()
        print(f"TRT engine loaded: {len(self.output_names)} outputs, max_batch={max_batch}")

    def __call__(self, x):
        B = x.shape[0]
        if x.dtype != self.input_dtype:
            x = x.to(self.input_dtype)
        with torch.cuda.stream(self._stream):
            self.context.set_input_shape(self.input_name, (B, *x.shape[1:]))
            self.context.set_tensor_address(self.input_name, x.data_ptr())
            for name, buf in zip(self.output_names, self._out_bufs):
                self.context.set_tensor_address(name, buf.data_ptr())
            self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        return [buf[:B].clone() for buf in self._out_bufs]


def gpu_seg_overlay(frames_gpu, da_masks, ll_masks):
    green = torch.tensor([0, 255, 0],  device=frames_gpu.device, dtype=torch.float32)
    blue  = torch.tensor([255, 0, 0],  device=frames_gpu.device, dtype=torch.float32)
    da_3d = (da_masks == 1).unsqueeze(-1)
    ll_3d = (ll_masks == 1).unsqueeze(-1)
    color = torch.zeros_like(frames_gpu)
    color = torch.where(da_3d, green.view(1,1,1,3).expand_as(color), color)
    color = torch.where(ll_3d, blue.view(1,1,1,3).expand_as(color),  color)
    has_color = color.mean(dim=-1, keepdim=True) != 0
    blended = torch.where(has_color, frames_gpu * 0.5 + color * 0.5, frames_gpu)
    return blended.byte()


def process(opt):
    device = torch.device(f'cuda:{opt.device}')
    os.makedirs(opt.save_dir, exist_ok=True)

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    engine  = TRTEngine(opt.engine, device, max_batch=opt.batch_size)
    dataset      = LoadImages(opt.source, img_size=opt.img_size)
    _frames_done = [0]  # shared counter filled by inference_loop
    print(f"  Processing {opt.source}\n")

    decode_queue = Queue(maxsize=8)
    save_queue   = Queue(maxsize=8)

    def decode_thread():
        buf_imgs, buf_meta = [], []
        for path, input_img, img0, vid_cap, shapes in dataset:
            h_lb, w_lb = input_img.shape[:2]  # letterboxed content dims (may not be square)
            t = torch.from_numpy(
                    np.ascontiguousarray(input_img)
                ).permute(2,0,1).float() / 255.0
            # Pad to square so TRT always gets the fixed img_size x img_size it was built for.
            # Content stays in the top-left; we track h_lb/w_lb to crop outputs correctly.
            if h_lb != opt.img_size or w_lb != opt.img_size:
                t = F.pad(t, (0, opt.img_size - w_lb, 0, opt.img_size - h_lb))
            img0_small = cv2.resize(img0, (w_lb, h_lb))
            fps_cap    = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30.0
            buf_meta.append((path, img0_small, shapes, vid_cap, h_lb, w_lb, fps_cap))
            buf_imgs.append(t)
            if len(buf_imgs) == opt.batch_size:
                decode_queue.put((torch.stack(buf_imgs), buf_meta.copy()))
                buf_imgs.clear(); buf_meta.clear()
        if buf_imgs:
            decode_queue.put((torch.stack(buf_imgs), buf_meta.copy()))
        decode_queue.put(SENTINEL)
        print("\n[decode] done")

    def inference_loop():
        frames_done = 0
        t_start = time.time()

        while True:
            item = decode_queue.get()
            if item is SENTINEL:
                save_queue.put(SENTINEL)
                break

            batch_cpu, meta = item
            N = len(meta)

            batch_gpu = batch_cpu.to(device).float()
            batch_gpu = (batch_gpu - MEAN) / STD
            H, W = batch_gpu.shape[2], batch_gpu.shape[3]

            with torch.no_grad():
                trt_outs = engine(batch_gpu)
                # outputs match export order: det_preds, da_seg, ll_seg
                det_preds_raw = trt_outs[0].float()   # [B, N, 6]
                da_seg_out    = trt_outs[1].float()   # [B, 2, H, W]
                ll_seg_out    = trt_outs[2].float()   # [B, 2, H, W]

                det_preds = non_max_suppression(
                    det_preds_raw,
                    conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
                    classes=None, agnostic=False
                )
                # Move to CPU immediately — save_thread must not touch GPU tensors,
                # since CUDA context corruption would crash it too.
                det_preds = [d.cpu() for d in det_preds]

                pads   = [(int(m[2][1][1][0]), int(m[2][1][1][1])) for m in meta]
                pad_w, pad_h = pads[0]
                out_h, out_w = meta[0][1].shape[:2]
                out_size = (out_h, out_w)

                # h_lb/w_lb is the actual letterboxed content size within the padded 640x640 input.
                # Crop to content region first, then remove letterbox padding.
                h_lb, w_lb = meta[0][4], meta[0][5]
                da_pred = da_seg_out[:, :, pad_h:(h_lb-pad_h), pad_w:(w_lb-pad_w)]
                ll_pred = ll_seg_out[:, :, pad_h:(h_lb-pad_h), pad_w:(w_lb-pad_w)]
                da_up = F.interpolate(da_pred, size=out_size, mode='bilinear')
                ll_up = F.interpolate(ll_pred, size=out_size, mode='bilinear')

                _, da_masks = torch.max(da_up, 1)
                _, ll_masks = torch.max(ll_up, 1)
                da_masks = (da_masks.int() - ll_masks.int()).clamp(min=0)
                ll_masks = ll_masks.int()

                img0s_np   = np.stack([m[1] for m in meta])
                img0s_np   = img0s_np[..., ::-1].copy()
                frames_gpu = torch.from_numpy(img0s_np).to(device).float()
                blended    = gpu_seg_overlay(frames_gpu, da_masks, ll_masks)
                blended_cpu = blended.cpu().numpy()

            save_queue.put((det_preds, blended_cpu, meta, (h_lb, w_lb)))
            frames_done += len(meta)
            elapsed = time.time() - t_start
            print(f"\r[inference] {frames_done} frames  {frames_done/elapsed:.1f} fps",
                  end='', flush=True)

        elapsed = time.time() - t_start
        _frames_done[0] = frames_done
        print(f"\n[inference] done — {frames_done} frames in {elapsed:.2f}s  "
              f"({frames_done/elapsed:.1f} fps)")

    def save_thread():
        vid_path, vid_writer = None, None
        frames_done = 0
        t_start = time.time()

        try:
            while True:
                item = save_queue.get()
                if item is SENTINEL:
                    break
                det_preds, blended_cpu, meta, (H, W) = item

                for i, (path, img0, shapes, vid_cap, h_lb, w_lb, fps_cap) in enumerate(meta):
                    img_det = blended_cpu[i]
                    det     = det_preds[i]
                    if len(det):
                        det[:, :4] = scale_coords((H, W), det[:, :4], img_det.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            if any(x != x for x in xyxy):  # skip NaN coords
                                continue
                            plot_one_box(xyxy, img_det, label=f'{conf:.2f}',
                                         color=(0,255,255), line_thickness=2)
                    save_path = str(Path(opt.save_dir) / Path(path).name)
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        h_o, w_o = img_det.shape[:2]
                        fps_out   = fps_cap if fps_cap > 0 else 30.0
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_out, (w_o, h_o))
                        if not vid_writer.isOpened():
                            print(f"[save] ERROR: VideoWriter failed to open {save_path}")
                    vid_writer.write(img_det)
                    frames_done += 1
        except Exception as e:
            print(f"[save] EXCEPTION: {e}")
            import traceback; traceback.print_exc()
        finally:
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()
        elapsed = time.time() - t_start
        print(f"[save] done — {frames_done} frames in {elapsed:.2f}s  "
              f"({frames_done/elapsed:.1f} fps)")

    t_total = time.time()
    t1 = threading.Thread(target=decode_thread, daemon=True)
    t3 = threading.Thread(target=save_thread,   daemon=True)
    t1.start(); t3.start()
    try:
        inference_loop()
    except Exception as e:
        print(f"\n[inference] CRASHED: {e} — flushing save thread")
        save_queue.put(SENTINEL)   # let save_thread release vid_writer before process exits
    t1.join()
    t3.join()  # save_thread will release vid_writer in its finally block
    wall         = time.time() - t_total
    total_frames = _frames_done[0]
    e2e_fps      = total_frames / wall if wall > 0 else 0.0
    print(f"\nWall time: {wall:.1f}s  ({total_frames} frames, {e2e_fps:.1f} fps end-to-end)")

    # Explicit cleanup: destroy TRT objects while CUDA context is still alive.
    # Without this, Python's GC tears down PyTorch's CUDA context first and
    # TRT's destructors fire against an already-dead context (cu13 vs cu124 mismatch).
    torch.cuda.synchronize(device)
    engine._out_bufs.clear()
    del engine.context
    del engine.engine
    torch.cuda.empty_cache()

    if opt.log:
        log_path   = Path(opt.log)
        write_header = not log_path.exists()
        with open(log_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'timestamp','source','engine','device',
                'batch_size','img_size','frames','wall_s','e2e_fps'
            ])
            if write_header:
                w.writeheader()
            w.writerow({
                'timestamp':  time.strftime('%Y-%m-%d %H:%M:%S'),
                'source':     opt.source,
                'engine':     opt.engine,
                'device':     opt.device,
                'batch_size': opt.batch_size,
                'img_size':   opt.img_size,
                'frames':     total_frames,
                'wall_s':     round(wall, 2),
                'e2e_fps':    round(e2e_fps, 1),
            })
        print(f"Speed log appended → {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine',     type=str,   default='weights/yolopx_fp16.trt')
    parser.add_argument('--source',     type=str,   default='inputs/')
    parser.add_argument('--save-dir',   type=str,   default='outputs/')
    parser.add_argument('--img-size',   type=int,   default=640)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres',  type=float, default=0.45)
    parser.add_argument('--device',     type=str,   default='0')
    parser.add_argument('--log',        type=str,   default='', help='CSV speed log path')
    opt = parser.parse_args()
    process(opt)
