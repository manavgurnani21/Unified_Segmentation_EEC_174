import onnxruntime as ort
import numpy as np
import os, sys, glob
import time
import cv2
import argparse
from pathlib import Path

# Auto-fix LD_LIBRARY_PATH for ONNXRuntime CUDAExecutionProvider
nvidia_libs = glob.glob(os.path.join(os.path.dirname(sys.executable), '../lib/python*/site-packages/nvidia/*/lib'))
if nvidia_libs and nvidia_libs[0] not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_libs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

def preprocess_frame(frame, img_size):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return img.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='YOLOPX/weights/yolopx_qat_epoch_0_sim.onnx')
    parser.add_argument('--source', type=str, default='YOLOPX/demo/2.gif')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=None)
    opt = parser.parse_args()

    print(f"=== Loading ONNX model in ONNXRuntime (CUDA) ===")
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(opt.onnx, providers=providers)
    input_name = session.get_inputs()[0].name

    print(f"=== Preprocessing frames ===")
    cap = cv2.VideoCapture(opt.source)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(preprocess_frame(frame, opt.img_size))
        count += 1
        if opt.max_frames and count >= opt.max_frames: break
    cap.release()

    if len(frames) == 0:
        print("No frames loaded.")
        return
    print(f"  Done — {len(frames)} frames loaded")
    
    # Stack frames
    all_frames = np.stack(frames)

    print("=== Warming up ===")
    warmup_batch = all_frames[:opt.batch_size]
    # Pad warmup batch if we don't have enough frames
    if warmup_batch.shape[0] < opt.batch_size:
        pad_size = opt.batch_size - warmup_batch.shape[0]
        warmup_batch = np.pad(warmup_batch, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
    _ = session.run(None, {input_name: warmup_batch})
    
    print(f"=== Running batched inference (Batch Size: {opt.batch_size}) ===")
    total_frames = len(frames)
    start = time.time()
    
    processed_frames = 0
    for i in range(0, total_frames, opt.batch_size):
        batch = all_frames[i:i+opt.batch_size]
        
        # Skip the last batch if it doesn't match the required static batch size
        if batch.shape[0] != opt.batch_size:
            print(f"  Skipping last batch of size {batch.shape[0]} (requires strict batch size {opt.batch_size})")
            continue
            
        _ = session.run(None, {input_name: batch})
        processed_frames += batch.shape[0]
        
    elapsed = time.time() - start

    print(f"\n=== Results (ONNXRuntime INT8 QDQ) ===")
    print(f"  Frames      : {processed_frames} (out of {total_frames})")
    print(f"  Batch size  : {opt.batch_size}")
    print(f"  Time        : {elapsed:.2f}s")
    print(f"  Pure GPU FPS: {total_frames / elapsed:.1f}")
    print(f"  ms/frame    : {1000 / (total_frames / elapsed):.1f}")

if __name__ == "__main__":
    main()