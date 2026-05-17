import argparse
import torch
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models import get_net
from lib.config import cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/epoch-195.pth', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    args = parser.parse_args()

    print("=== Loading PyTorch Blueprint ===")
    
    # We must explicitly disable tracing/autograd for export
    model = get_net(cfg)
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Create dummy input (this tells ONNX the shape of our input data)
    # Ensure it's explicitly marked as float32
    dummy_input = torch.randn(args.batch_size, 3, args.img_size[0], args.img_size[1], dtype=torch.float32)

    # Export to ONNX
    onnx_path = args.weights.replace('.pth', '.onnx')
    print(f"=== Exporting to ONNX: {onnx_path} ===")
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            verbose=False,
            opset_version=18,
            input_names=['images'],
            output_names=['det_out', 'drive_area_out', 'lane_line_out']
        )
    
    print("Export successful!")

if __name__ == '__main__':
    main()