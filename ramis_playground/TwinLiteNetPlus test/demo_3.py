# another gemini script to push the limits of the GPU without a CPU bottleneck in the way.
# gets about 14ms frame times (~70fps)

import torch
import torch.backends.cudnn as cudnn
import time
from model.model import TwinLiteNetPlus

class DummyArgs:
    def __init__(self, config):
        self.config = config

def run_benchmark(config_type='large', h=384, w=640):
    device = torch.device("cuda:0")
    cudnn.benchmark = True 
    
    args = DummyArgs(config=config_type)

    # 1. Setup Model
    model = TwinLiteNetPlus(args) 
    model.to(device).eval().half() 

    # 2. Create Dummy Data (Corrected Size Unpacking)
    # Dimensions: [Batch, Channels, Height, Width]
    input_size = (1, 3, h, w)
    dummy_input = torch.randn(*input_size).to(device).half()

    # 3. Warmup
    print(f"Warming up GPU with {config_type} model at {h}x{w}...")
    for _ in range(50):
        _ = model(dummy_input)

    # 4. Timing Loop
    num_frames = 1000
    torch.cuda.synchronize() 
    start_time = time.time()

    for _ in range(num_frames):
        _ = model(dummy_input)

    torch.cuda.synchronize() 
    end_time = time.time()

    # 5. Results
    total_time = end_time - start_time
    fps = num_frames / total_time
    ms_per_frame = (total_time / num_frames) * 1000

    print(f"\n--- {config_type.upper()} MODEL BENCHMARK ---")
    print(f"Resolution: {w}x{h} (Width x Height)")
    print(f"Average Latency: {ms_per_frame:.2f}ms")
    print(f"Raw GPU Speed: {fps:.2f} FPS")

if __name__ == '__main__':
    with torch.no_grad():
        # Using 384 (Height) x 640 (Width)
        run_benchmark(config_type='large', h=384, w=640)