# tools/gpu_benchmark.py
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models import get_net
from lib.config import cfg

# ---- CONFIG ----
VIDEO_PATH = "inputs/dashcam_freeway.mp4"
WEIGHTS    = "weights/epoch-195.pth"
IMG_SIZE   = 640
BATCH_SIZE = 8      # lower if you get OOM (try 4, 2, 1)
MAX_FRAMES = None    # set to None to use entire video
DEVICE     = torch.device("cuda:0")
# ----------------

def preprocess_frame(frame, img_size):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).float() / 255.0

print("=== Loading model ===")
model = get_net(cfg)
checkpoint = torch.load(WEIGHTS, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(DEVICE)
model.half()   # FP16
model.eval()

print("=== Preprocessing frames (CPU) ===")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(preprocess_frame(frame, IMG_SIZE))
    count += 1
    if MAX_FRAMES and count >= MAX_FRAMES:
        break
    if count % 50 == 0:
        print(f"  Loaded {count} frames...")
cap.release()

print(f"  Done — {len(frames)} frames loaded")

# Stack and move entire buffer to GPU at once
all_frames = torch.stack(frames).half().to(DEVICE)  # (N, 3, H, W)
print(f"  Buffer on GPU: {all_frames.shape}, {all_frames.nbytes / 1e6:.1f} MB")

# Warmup pass (important — first inference is always slow due to CUDA JIT)
print("=== Warming up ===")
with torch.no_grad():
    _ = model(all_frames[:2])
torch.cuda.synchronize()

print("=== Running batched inference ===")
total_frames = len(frames)
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    for i in range(0, total_frames, BATCH_SIZE):
        batch = all_frames[i:i+BATCH_SIZE]
        _ = model(batch)

torch.cuda.synchronize()  # wait for all GPU ops to finish before stopping timer
elapsed = time.time() - start

print(f"\n=== Results ===")
print(f"  Frames      : {total_frames}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Time        : {elapsed:.2f}s")
print(f"  Pure GPU FPS: {total_frames / elapsed:.1f}")
print(f"  ms/frame    : {1000 / (total_frames / elapsed):.1f}")
