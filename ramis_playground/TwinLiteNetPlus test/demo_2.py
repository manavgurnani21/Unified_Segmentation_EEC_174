# gemini script to test the GPU speed, as the prepackaged demo is CPU bottlenecked on my end. i went from 8fps with the CPU bottleneck, 
# to 52fps, though that also seems to be a CPU bottleneck. However, the 52fps bottleneck looks like one that cannot be overcome on prod.
# run with 
# python demo_2.py --weight 'weights/medium.pth' --source 'inputs/dashcam_rural.mp4' --config 'medium' --img-size 640
import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import time
from tqdm import tqdm
import numpy as np

from model.model import TwinLiteNetPlus
from demoDataset import LoadImages

def speed_test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True 
    
    # 1. Model Setup
    model = TwinLiteNetPlus(args)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.to(device).eval().half() 

    # 2. Dataset
    dataset = LoadImages(args.source, img_size=args.img_size)
    
    # 3. Warmup
    print("Warming up GPU...")
    dummy_input = torch.zeros((1, 3, args.img_size, args.img_size), device=device).half()
    for _ in range(20):
        _ = model(dummy_input)

    # 4. Pure Inference Loop
    print(f"Starting Speed Test on {len(dataset)} frames...")
    torch.cuda.synchronize()
    start_time = time.time()

    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        # --- FIX FOR TYPEERROR ---
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(device)
        else:
            img = img.to(device)
        
        img = img.half() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # GPU ONLY: No saving, no drawing, no upscaling
        with torch.no_grad():
            _ = model(img)
            
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    fps = len(dataset) / total_time
    print(f"\n--- RESULTS ---")
    print(f"Total Frames: {len(dataset)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Pure Inference Speed: {fps:.2f} FPS")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='pretrained/large.pth')
    parser.add_argument('--source', type=str, default='inference/videos')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--config', type=str, default="large")
    opt = parser.parse_args()

    speed_test(opt)