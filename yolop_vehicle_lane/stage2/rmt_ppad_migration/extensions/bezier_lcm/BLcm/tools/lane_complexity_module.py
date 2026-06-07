"""Phase BLcm: the Lane Complexity Module.

Architectural twin of GCA's (task_adapter_seg + gate_seg) pair, but for
curve complexity instead of task-feature blending.

Inputs:
    pooled_features: (B, P, fc_dim) post-ROIGather features.
Outputs (dict):
    degree_weights:   (B, P, 3) softmax over {K=1, K=2, K=3}, clamped
                      to [clamp_min, clamp_max] then renormalized.
    complexity_score: scalar mean effective K (1..3). For monitoring,
                      parallel to GCA's `gate_mean`.

LCM's first layer is bias-initialized to prefer K=3 (full cubic) so the
model starts from the safest geometric mode and learns to deviate as the
complexity penalty engages.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexityFeatureAdapter(nn.Module):
    """Tiny MLP that prepares pooled curve features for the gate."""

    def __init__(self, in_dim: int = 64, mid_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.ln1 = nn.LayerNorm(mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)), inplace=False)
        x = F.relu(self.ln2(self.fc2(x)), inplace=False)
        return x


class DegreeGate(nn.Module):
    """Outputs (clamped) softmax weights over Bezier degrees {K=1, K=2, K=3}."""

    def __init__(
        self, in_dim: int = 16, mid_dim: int = 8,
        clamp_min: float = 0.02, clamp_max: float = 0.96,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, 3)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self._init_weights()

    def _init_weights(self):
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Bias K=3 logit by +1.0 so softmax starts with K=3 preferred:
        # softmax([0, 0, 1]) ~= [0.21, 0.21, 0.58]. Matches GCA's "start
        # from default behavior, learn to deviate" pattern.
        with torch.no_grad():
            self.fc2.bias[2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x), inplace=False)
        logits = self.fc2(h)
        w = F.softmax(logits, dim=-1)
        # Clamp + renormalize so no single K dominates 100% (mirrors GCA's
        # clamp(0.05, 0.95) gate). Renormalize is essential - without it
        # the clamped tensor no longer sums to 1 and the mixture renderer
        # produces over- / under-scaled curves.
        w = w.clamp(self.clamp_min, self.clamp_max)
        w = w / w.sum(dim=-1, keepdim=True)
        return w


class LaneComplexityModule(nn.Module):
    """Composed module: adapter -> gate, with monitoring scalar."""

    def __init__(
        self,
        feat_dim: int = 64,
        adapter_mid_dim: int = 32,
        adapter_out_dim: int = 16,
        gate_mid_dim: int = 8,
        clamp_min: float = 0.02,
        clamp_max: float = 0.96,
    ):
        super().__init__()
        self.adapter = ComplexityFeatureAdapter(
            in_dim=feat_dim, mid_dim=adapter_mid_dim, out_dim=adapter_out_dim,
        )
        self.gate = DegreeGate(
            in_dim=adapter_out_dim, mid_dim=gate_mid_dim,
            clamp_min=clamp_min, clamp_max=clamp_max,
        )
        # K values cached so we don't recreate the tensor each forward
        self.register_buffer(
            'K_values', torch.tensor([1.0, 2.0, 3.0]), persistent=False,
        )

    def forward(self, pooled_features: torch.Tensor) -> dict:
        """
        Args:
            pooled_features: (B, P, feat_dim) features along each prior.
        Returns:
            dict with keys:
                degree_weights:  (B, P, 3)
                complexity_score: scalar tensor (detached)
        """
        adapted = self.adapter(pooled_features)
        weights = self.gate(adapted)  # (B, P, 3)
        effective_K = (weights * self.K_values).sum(dim=-1)  # (B, P)
        complexity_score = effective_K.mean().detach()
        return {
            'degree_weights': weights,
            'complexity_score': complexity_score,
            'effective_K': effective_K,  # kept (non-detached) for any loss term that wants it
        }


def _smoke_test() -> int:
    """Run the module on a random tensor and verify shape / range."""
    torch.manual_seed(0)
    lcm = LaneComplexityModule(feat_dim=64)
    pooled = torch.randn(2, 192, 64)
    out = lcm(pooled)
    w = out['degree_weights']
    assert w.shape == (2, 192, 3), f'unexpected shape {w.shape}'
    assert torch.all(w >= 0.02 - 1e-6)
    assert torch.all(w <= 0.96 + 1e-6)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        'degree weights must renormalize to 1'
    print(f'[BLcm.smoke] weights mean = {w.mean(dim=(0,1))}')
    print(f'[BLcm.smoke] complexity_score = {out["complexity_score"]:.3f}')
    # With the K=3 bias of +1 we expect effective_K well above 2 at init.
    assert out['complexity_score'] > 2.0, 'init bias should favor K=3'
    print('[BLcm.smoke] OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(_smoke_test())


__all__ = [
    'ComplexityFeatureAdapter', 'DegreeGate', 'LaneComplexityModule',
]
