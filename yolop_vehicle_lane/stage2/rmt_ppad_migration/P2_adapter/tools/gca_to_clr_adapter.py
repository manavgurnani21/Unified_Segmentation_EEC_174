"""Phase P2: GCA -> CLRHead channel adapter.

Bridges RMT-PPAD's GCA-task-adapted segmentation feature pyramid (256-channel,
1:1 spatial ratio) to the channel count CLRHead expects (64-channel, same
spatial ratio).

Implemented per appendix-path3-implementation-prompt.md sec 5.2: three
independent 1x1 conv -> BN -> ReLU blocks, one per pyramid level. Spatial
resolution is preserved at each level - aspect-ratio handling (the other
mismatch between GCA's 1:1 output and CLRHead's CULane 1:2.5 expectation)
is done in P3 by re-initializing CLRHead's priors for square images.

Per the hard constraint in the migration README we do NOT modify
external_repos/RMT-PPAD-main; this lives as a standalone module that P4
will import and inject into the vendored MTDETRDecoder.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GCAtoCLRAdapter(nn.Module):
    """3-level 256->64 channel projection for the lane-head adapter chain.

    Input  (x_proj_seg from MTDETRDecoder's post-GCA stream):
        list of 3 tensors [X3, F4, F5]
        where X3 is (B, 256, 80, 80), F4 is (B, 256, 40, 40),
        F5 is (B, 256, 20, 20) for the canonical 640x640 input.

    Output:
        list of 3 tensors with the same spatial shapes, 64 channels each.

    Each level uses an independent (Conv1x1 -> BN -> ReLU) block; weights
    are Kaiming-normal initialized for the ReLU regime.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 64,
                 num_levels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels)
        ])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_proj_seg: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(x_proj_seg) != self.num_levels:
            raise ValueError(
                f'GCAtoCLRAdapter expected {self.num_levels} pyramid levels, '
                f'got {len(x_proj_seg)}.'
            )
        return [self.proj[i](x_proj_seg[i]) for i in range(self.num_levels)]


def smoke_test() -> int:
    """Acceptance test from appendix-path3-implementation-prompt.md sec 5.3,
    plus a backward-pass check and a parameter-count sanity check."""
    import sys
    print('[smoke] GCAtoCLRAdapter starting', flush=True)

    adapter = GCAtoCLRAdapter(in_channels=256, out_channels=64)
    print(f'[smoke] params = {sum(p.numel() for p in adapter.parameters()):,}', flush=True)

    fake_input = [
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]
    out = adapter(fake_input)

    expected_shapes = [(2, 64, 80, 80), (2, 64, 40, 40), (2, 64, 20, 20)]
    if len(out) != 3:
        print(f'[smoke] FAIL: expected 3 output tensors, got {len(out)}')
        return 1
    for i, (got, want) in enumerate(zip(out, expected_shapes)):
        print(f'[smoke] level {i}: in {tuple(fake_input[i].shape)} -> '
              f'out {tuple(got.shape)}  (expect {want})')
        if tuple(got.shape) != want:
            print(f'[smoke] FAIL: shape mismatch at level {i}')
            return 1

    # Backward-pass sanity: gradients should flow through every projection.
    adapter.train()
    out = adapter(fake_input)
    loss = sum(t.mean() for t in out)
    loss.backward()
    for i, block in enumerate(adapter.proj):
        conv = block[0]
        if conv.weight.grad is None or conv.weight.grad.abs().sum() == 0:
            print(f'[smoke] FAIL: level {i} got no gradient')
            return 1
    print('[smoke] backward pass: all 3 projections received non-zero gradient')

    # Wrong-length input should raise.
    try:
        adapter([fake_input[0], fake_input[1]])
    except ValueError as e:
        print(f'[smoke] wrong-length input correctly raised: {e}')
    else:
        print('[smoke] FAIL: expected ValueError for 2-element input')
        return 1

    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(smoke_test())
