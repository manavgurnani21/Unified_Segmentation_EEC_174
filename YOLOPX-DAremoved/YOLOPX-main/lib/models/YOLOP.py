# FILE: lib/models/YOLOP.py

import torch
from torch import tensor
import torch.nn as nn

from torch.nn import Conv2d

import sys,os
import math
import sys
sys.path.append(os.getcwd())

from lib.utils import initialize_weights
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.common import Conv, seg_head, PSA_p, MergeBlock
from lib.models.common import Concat, FPN_C2, FPN_C3, FPN_C4, ELANNet, ELANBlock_Head, PaFPNELAN, IDetect, RepConv
from lib.models.YOLOX_Head_scales_noshare import YOLOXHead

YOLOP = [

[2, 19],

[ -1, ELANNet, [True]],

[ -1, PaFPNELAN, []],

[ -1, YOLOXHead,  [1]],

[ 1, FPN_C3, []],

[ 1, FPN_C4, []],

[ 1, FPN_C2, []],

[ -1, Conv, [256, 128, 3, 1]],

[ 3, Conv, [256, 128, 3, 1]],

[ -1, Upsample, [None, 2, 'bilinear']],

[ [-1, 6], MergeBlock, ["add"]],

[ -1, ELANBlock_Head, [128, 64]],

[ -1, PSA_p, [64, 64]],

[ -1, Conv, [64, 32, 3, 1]],

[ -1, Upsample, [None, 2, 'bilinear']],

[ -1, Conv, [32, 16, 3, 1]],

[ -1, ELANBlock_Head, [16, 8]],

[ -1, PSA_p, [8, 8]],

[ -1, Upsample, [None, 2, 'bilinear']],

[ -1, Conv, [8, 2, 3, 1]],

[ -1, seg_head, ['sigmoid']],

]

class MCnet(nn.Module):

    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1

        self.det_out_idx = block_cfg[0][0]

        self.seg_out_idx = block_cfg[0][1:]

        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block

            if block is YOLOXHead:
                self.detector_index = i

            block_ = block(*args)

            block_.index, block_.from_ = i, from_

            layers.append(block_)

            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)

        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)

        self.names = [str(i) for i in range(self.nc)]

        Detector = self.model[self.detector_index]

        if isinstance(Detector, YOLOXHead):

            s = 512

            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))

            self.stride = Detector.strides

            Detector.initialize_biases(1e-2)

        initialize_weights(self)

    def forward(self, x):

        cache = []

        out = []

        det_out = None

        for i, block in enumerate(self.model):

            if block.from_ != -1:

                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]

            x = block(x)

            if i in self.seg_out_idx:
                out.append(x)

            if i == self.detector_index:
                det_out = x

            cache.append(x if block.index in self.save else None)

        out.insert(0,det_out)

        return out

    def fuse(self):

        print('Fusing layers... ')

        for m in self.model.modules():

            if isinstance(m, RepConv):

                m.fuse_repvgg_block()

            elif type(m) is Conv and hasattr(m, 'bn'):

                m.conv = fuse_conv_and_bn(m.conv, m.bn)

                delattr(m, 'bn')

                m.forward = m.fuseforward

            elif isinstance(m, IDetect):

                m.fuse()

                m.forward = m.fuseforward

        return self

def get_net(cfg, **kwargs):

    m_block_cfg = YOLOP

    model = MCnet(m_block_cfg, **kwargs)

    return model

def fuse_conv_and_bn(conv, bn):

    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)

    w_bn = torch.diag(
        bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    )

    fusedconv.weight.copy_(
        torch.mm(w_bn, w_conv).view(fusedconv.weight.shape)
    )

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias

    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))

    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    )

    return fusedconv

if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter

    model = get_net(False)

    input_ = torch.randn((1, 3, 256, 256))

    gt_ = torch.rand((1, 2, 256, 256))

    metric = SegmentationMetric(2)

    model_out = model(input_)

    detects, lane_line_seg = model_out

    for det in detects:
        print(det.shape)

    print(lane_line_seg.shape)