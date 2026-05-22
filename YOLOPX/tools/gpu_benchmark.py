# tools/gpu_benchmark.py
import sys
import time
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0)}")

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import sys
if '--precision' in sys.argv and 'int8' in sys.argv:
    from pytorch_quantization import quant_modules
    quant_modules.initialize()
from lib.models import get_net
from lib.config import cfg
def preprocess_frame(frame, img_size):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).float() / 255.0

def main(opt):
    DEVICE = torch.device(opt.device)
    
    print(f"=== Loading model in {opt.precision.upper()} ===")
    model = get_net(cfg)
    
    # Try to load weights if they exist
    if Path(opt.weights).exists():
        checkpoint = torch.load(opt.weights, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"Warning: Weights {opt.weights} not found. Benchmarking with random weights.")
        
    model = model.to(DEVICE)
    
    # *** QUANTIZATION TOGGLE ***
    if opt.precision == "fp16":
        model.half()   # Convert model weights to 16-bit half precision
    else:
        model.float()  # Ensure model weights are 32-bit full precision (default)
        
    model.eval()

    print(f"=== Preprocessing frames (CPU) ===")
    if not Path(opt.source).exists():
        print(f"Error: Video file {opt.source} not found! Please provide a valid --source")
        return
        
    cap = cv2.VideoCapture(opt.source)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, opt.img_size))
        count += 1
        if opt.max_frames and count >= opt.max_frames:
            break
        if count % 50 == 0:
            print(f"  Loaded {count} frames...")
    cap.release()

    if len(frames) == 0:
        print("No frames loaded.")
        return

    print(f"  Done — {len(frames)} frames loaded")

    # Stack and move entire buffer to GPU at once
    all_frames = torch.stack(frames)
    
    # *** DATA QUANTIZATION TOGGLE ***
    if opt.precision == "fp16":
        all_frames = all_frames.half() # Convert input data to FP16
    else:
        all_frames = all_frames.float() # Keep input data as FP32
        
    all_frames = all_frames.to(DEVICE)  # (N, 3, H, W)
    print(f"  Buffer on GPU: {all_frames.shape}, {all_frames.nbytes / 1e6:.1f} MB")

    # Warmup pass (important — first inference is always slow due to CUDA JIT)
    print("=== Warming up ===")
    with torch.no_grad():
        _ = model(all_frames[:2])
    torch.cuda.synchronize()

    print(f"=== Running batched inference (Batch Size: {opt.batch_size}) ===")
    total_frames = len(frames)
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for i in range(0, total_frames, opt.batch_size):
            batch = all_frames[i:i+opt.batch_size]
            _ = model(batch)

    torch.cuda.synchronize()  # wait for all GPU ops to finish before stopping timer
    elapsed = time.time() - start

    print(f"\n=== Results ({opt.precision.upper()}) ===")
    print(f"  Frames      : {total_frames}")
    print(f"  Batch size  : {opt.batch_size}")
    print(f"  Time        : {elapsed:.2f}s")
    print(f"  Pure GPU FPS: {total_frames / elapsed:.1f}")
    print(f"  ms/frame    : {1000 / (total_frames / elapsed):.1f}")
    
    # Calculate and print peak VRAM
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    print(f"  Peak VRAM   : {peak_vram:.1f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/epoch-195.pth', help='model.pth path')
    parser.add_argument('--source', type=str, default='demo/2.gif', help='video/gif path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for benchmarking')
    parser.add_argument('--max-frames', type=int, default=None, help='max frames to benchmark')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32', help='Precision mode')
    opt = parser.parse_args()
    
    main(opt)
