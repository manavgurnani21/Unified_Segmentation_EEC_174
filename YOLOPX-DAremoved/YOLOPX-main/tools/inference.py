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


def process(cfg, opt):

    device = select_device(None, opt.device)

    half = device.type != 'cpu'

    os.makedirs(opt.save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    MEAN = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    ).view(1,3,1,1).half()

    STD = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    ).view(1,3,1,1).half()

    print("=== Loading model ===")

    model = get_net(cfg)

    ckpt = torch.load(
        opt.weights,
        map_location=device
    )

    model_dict = model.state_dict()

    pretrained_dict = {
        k: v
        for k, v in ckpt['state_dict'].items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    model = model.to(device)

    if half:
        model.half()

    dummy = torch.zeros(
        (1, 3, opt.img_size, opt.img_size),
        device=device
    )

    _ = model(dummy.half() if half else dummy)

    model.eval()

    dataset = LoadImages(
        opt.source,
        img_size=opt.img_size
    )

    total = len(dataset)

    print(f"{total} frames to process")

    decode_queue = Queue(maxsize=8)

    save_queue = Queue(maxsize=8)

    def decode_thread():

        buf_imgs = []

        buf_meta = []

        for path, input_img, img0, vid_cap, shapes in dataset:

            t = torch.from_numpy(
                np.ascontiguousarray(input_img)
            ).permute(2,0,1).float() / 255.0

            buf_imgs.append(t)

            out_h, out_w = input_img.shape[:2]

            img0_small = cv2.resize(
                img0,
                (out_w, out_h)
            )

            buf_meta.append(
                (
                    path,
                    img0_small,
                    shapes,
                    vid_cap
                )
            )

            if len(buf_imgs) == opt.batch_size:

                decode_queue.put(
                    (
                        torch.stack(buf_imgs),
                        buf_meta.copy()
                    )
                )

                buf_imgs.clear()

                buf_meta.clear()

        if buf_imgs:

            decode_queue.put(
                (
                    torch.stack(buf_imgs),
                    buf_meta.copy()
                )
            )

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

            batch_gpu = batch_cpu.to(device).half()

            batch_gpu = (batch_gpu - MEAN) / STD

            H, W = batch_gpu.shape[2], batch_gpu.shape[3]

            torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():

                det_out, ll_seg_out = model(batch_gpu)

                inf_out, _ = det_out

                det_preds = non_max_suppression(
                    inf_out,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    classes=None,
                    agnostic=False
                )

                pads = [
                    (
                        int(m[2][1][1][0]),
                        int(m[2][1][1][1])
                    )
                    for m in meta
                ]

                pad_w, pad_h = pads[0]

                out_h, out_w = meta[0][1].shape[:2]

                out_size = (out_h, out_w)

                ll_pred = ll_seg_out[
                    :,
                    :,
                    pad_h:(H-pad_h),
                    pad_w:(W-pad_w)
                ]

                ll_up = F.interpolate(
                    ll_pred.float(),
                    size=out_size,
                    mode='bilinear'
                )

                ll_up = torch.softmax(
                    ll_up,
                    dim=1
                )

                lane_prob = ll_up[:, 1, :, :]

                ll_masks = (lane_prob > 0.90).int()

                img0s_np = np.stack(
                    [m[1] for m in meta]
                )

                img0s_np = np.ascontiguousarray(
                    img0s_np[..., ::-1]
                )

                blended_cpu = img0s_np.copy()

            torch.cuda.synchronize()

            end_time = time.time()

            fps = len(meta) / (end_time - start_time)

            print(f"\nFPS: {fps:.2f}")

            print(
                f"Inference Time: "
                f"{(end_time-start_time)*1000:.2f} ms"
            )

            save_queue.put(
                (
                    det_preds,
                    blended_cpu,
                    meta,
                    (H, W)
                )
            )

            frames_done += len(meta)

            elapsed = time.time() - t_start

            print(
                f"\r[inference] "
                f"{frames_done}/{total} "
                f"{frames_done/elapsed:.1f} fps",
                end='',
                flush=True
            )

        elapsed = time.time() - t_start

        print(
            f"\n[inference] done "
            f"{frames_done} frames "
            f"in {elapsed:.2f}s "
            f"({frames_done/elapsed:.1f} fps)"
        )

    def save_thread():

        vid_path = None

        vid_writer = None

        while True:

            item = save_queue.get()

            if item is SENTINEL:

                break

            det_preds, blended_cpu, meta, (H, W) = item

            for i, (
                path,
                img0,
                shapes,
                vid_cap
            ) in enumerate(meta):

                img_det = blended_cpu[i]

                det = det_preds[i]

                if len(det):

                    det[:, :4] = scale_coords(
                        (H, W),
                        det[:, :4],
                        img_det.shape
                    ).round()

                    for *xyxy, conf, cls in reversed(det):

                        plot_one_box(
                            xyxy,
                            img_det,
                            label=f'{conf:.2f}',
                            color=(0,255,255),
                            line_thickness=2
                        )

                save_path = str(
                    Path(opt.save_dir) / Path(path).name
                )

                if vid_path != save_path:

                    vid_path = save_path

                    if isinstance(
                        vid_writer,
                        cv2.VideoWriter
                    ):
                        vid_writer.release()

                    fps_out = (
                        vid_cap.get(cv2.CAP_PROP_FPS)
                        if vid_cap else 30
                    )

                    h_o, w_o = img_det.shape[:2]

                    vid_writer = cv2.VideoWriter(
                        save_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps_out,
                        (w_o, h_o)
                    )

                vid_writer.write(img_det)

        if isinstance(
            vid_writer,
            cv2.VideoWriter
        ):
            vid_writer.release()

        print(f"Output: {opt.save_dir}")

    t1 = threading.Thread(
        target=decode_thread,
        daemon=True
    )

    t3 = threading.Thread(
        target=save_thread,
        daemon=True
    )

    t1.start()

    t3.start()

    inference_loop()

    t1.join()

    t3.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--weights',
        type=str,
        default='weights/epoch-195.pth'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='inference/video/video1.mp4'
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='inference/video_output'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=512
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.3
    )

    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.45
    )

    parser.add_argument(
        '--device',
        type=str,
        default='0'
    )

    opt = parser.parse_args()

    process(cfg, opt)