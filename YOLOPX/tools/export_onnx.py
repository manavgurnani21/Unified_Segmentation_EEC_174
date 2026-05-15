# python tools/export_onnx.py --weights weights/epoch-195.pth --output weights/yolopx.onnx

import argparse, os, sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from lib.config import cfg
from lib.models import get_net


class _ONNXWrapper(nn.Module):
    """Flattens the model's nested tuple output to three plain tensors."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # out = [det_out, da_seg, ll_seg]
        # det_out = (decoded_preds [B,N,6], raw_scale_list)  when decode_in_inference=True
        det_out, da_seg, ll_seg = out
        det_preds = det_out[0]   # [B, N, 6]
        return det_preds, da_seg, ll_seg


def export(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Loading model ...")
    model = get_net(cfg)
    ckpt  = torch.load(opt.weights, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).float().eval()
    model.fuse()   # fuse Conv+BN layers

    wrapper = _ONNXWrapper(model).eval()

    dummy = torch.zeros(1, 3, opt.img_size, opt.img_size, device=device)
    with torch.no_grad():
        _ = wrapper(dummy)   # dry-run to catch shape errors early

    print(f"Exporting to {opt.output} ...")
    Path(opt.output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        opt.output,
        opset_version=17,
        input_names=['images'],
        output_names=['det_preds', 'da_seg', 'll_seg'],
        dynamic_axes={
            'images':    {0: 'batch'},
            'det_preds': {0: 'batch'},
            'da_seg':    {0: 'batch'},
            'll_seg':    {0: 'batch'},
        },
        do_constant_folding=True,
    )
    print("ONNX export done.")

    if opt.simplify:
        try:
            import onnx
            import onnxsim
            model_onnx = onnx.load(opt.output)
            model_onnx, ok = onnxsim.simplify(model_onnx)
            assert ok, "onnxsim returned failure"
            onnx.save(model_onnx, opt.output)
            print("onnxsim simplification done.")
        except ImportError:
            print("onnx / onnxsim not installed — skipping simplification.")
            print("  pip install onnx onnxsim")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',   default='weights/epoch-195.pth')
    parser.add_argument('--output',    default='weights/yolopx.onnx')
    parser.add_argument('--img-size',  type=int, default=640)
    parser.add_argument('--simplify',  action='store_true', help='run onnxsim after export')
    opt = parser.parse_args()
    export(opt)
