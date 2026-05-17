import argparse
import torch
import sys
from pathlib import Path

# Initialize quant modules BEFORE importing the model
from pytorch_quantization import quant_modules
import pytorch_quantization
quant_modules.initialize()

from pytorch_quantization.nn import TensorQuantizer
TensorQuantizer.use_fb_fake_quant = True

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models import get_net
from lib.config import cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='YOLOPX/weights/yolopx_qat_epoch_0.pth', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    args = parser.parse_args()

    print("=== Loading PyTorch Blueprint (QAT) ===")
    
    # We must explicitly disable tracing/autograd for export
    model = get_net(cfg)
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.img_size[0], args.img_size[1], dtype=torch.float32).cuda()
    model.cuda()

    # Disable quantizers on the final detection heads to avoid TRT Myelin crash
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if any(x in name for x in ['preds', 'det_out', 'drive_area_out', 'lane_line_out', 'model.2.', 'model.24', 'model.30', 'model.28', 'model.26']):
                module.disable_quant()

    # Export to ONNX
    onnx_path = args.weights.replace('.pth', '.onnx')
    print(f"=== Exporting QAT model to ONNX: {onnx_path} ===")
    
    import torch.onnx._internal.torchscript_exporter.utils as ts_utils
    with torch.no_grad():
        with pytorch_quantization.enable_onnx_export():
            ts_utils.export(
                model, 
                dummy_input, 
                onnx_path,
                verbose=False,
                opset_version=17, # 17 is recommended for newer TRT/PyTorch
                input_names=['images'],
                output_names=['det_out', 'drive_area_out', 'lane_line_out'],
                do_constant_folding=True
            )
    
    print("Export successful!")

if __name__ == '__main__':
    main()