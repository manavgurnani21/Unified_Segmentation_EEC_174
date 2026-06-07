"""Phase B4: Bezier head with optional LCM.

Drop-in replacement for `P4_lane_head_integration/tools/lane_seg_head.py`
that emits 16-D Bezier vectors instead of 78-D polylines.

Architecture:
    GCAtoCLRAdapter (reused from P2)                 # 256 -> 64 dims
        -> CLRHead backbone (with NEW reg_layers     # 64-D pooled feats
            outputting 12 deltas instead of 76)         per prior
        -> [optional] LaneComplexityModule           # 3 degree weights
    Returns dict {'lane_output': {...}}

Forward signature matches LaneSegHead so MTDETRDecoder can swap them.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_migration_root() -> Path:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            return cur
        cur = cur.parent
    for ancestor in here.parents:
        cand = ancestor / 'yolop_vehicle_lane' / 'stage2' / 'rmt_ppad_migration'
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate rmt_ppad_migration/ from {here}")


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


_MIG = _find_migration_root()
_p2 = _import_by_path(
    'b4_p2_adapter',
    _MIG / 'P2_adapter' / 'tools' / 'gca_to_clr_adapter.py',
)
_p3 = _import_by_path(
    'b4_p3_square_head',
    _MIG / 'P3_square_clrhead' / 'tools' / 'clr_head_square.py',
)
_lcm_mod = _import_by_path(
    'b4_blcm',
    _MIG / 'extensions' / 'bezier_lcm' / 'BLcm' / 'tools' / 'lane_complexity_module.py',
)

GCAtoCLRAdapter = _p2.GCAtoCLRAdapter  # type: ignore[attr-defined]
CLRHeadForSquareImage = _p3.CLRHeadForSquareImage  # type: ignore[attr-defined]
LaneComplexityModule = _lcm_mod.LaneComplexityModule  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helper: priors -> initial Bezier control points
# ---------------------------------------------------------------------------

def priors_to_bezier_init(priors: torch.Tensor) -> torch.Tensor:
    """Convert (..., 78) priors to (..., 8) Bezier control-point packing.

    Per appendix Bezier sec 5.4.
    Input prior fields used:
        priors[..., 2] = start_y   (normalized [0, 1])
        priors[..., 3] = start_x   (normalized [0, 1])
        priors[..., 4] = theta     (normalized [0, 1] = theta_rad/pi)
        priors[..., 5] = length    (strip count, ~ [0, 71])

    Output packing: [P0x, P1x, P2x, P3x, P0y, P1y, P2y, P3y].
    All in normalized [0, 1] x [0, 1] (target frame).
    """
    start_y = priors[..., 2]
    start_x = priors[..., 3]
    theta = priors[..., 4] * math.pi
    # Convert strip count to normalized length. n_strips=71 in CLRHead.
    length_norm = priors[..., 5].clamp(min=1.0) / 71.0

    P0_x = start_x
    P0_y = start_y
    P3_x = start_x + length_norm * torch.sin(theta)
    # Image y axis grows downward but priors store start_y as normalized
    # row-fraction from BOTTOM; sweep direction is upward, so subtract.
    P3_y = start_y - length_norm * torch.cos(theta)

    P1_x = (2.0 * P0_x + P3_x) / 3.0
    P1_y = (2.0 * P0_y + P3_y) / 3.0
    P2_x = (P0_x + 2.0 * P3_x) / 3.0
    P2_y = (P0_y + 2.0 * P3_y) / 3.0

    return torch.stack(
        [P0_x, P1_x, P2_x, P3_x, P0_y, P1_y, P2_y, P3_y], dim=-1,
    )


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class _AllowedKeysHasKey:
    """Picklable replacement for `lambda k, _a=allowed: k in _a`.

    Identical to the helper in lane_seg_head.py - duplicated locally so
    each head module is independently importable. The lambda version of
    this broke `strip_optimizer()` at end of training because its
    qualname includes `<locals>` and pickle/torch.save can't resolve
    local functions.
    """

    def __init__(self, allowed_keys):
        self._allowed = frozenset(allowed_keys)

    def __call__(self, key):
        return key in self._allowed


def _build_clr_cfg(img_size: int, max_lanes: int) -> SimpleNamespace:
    cfg = SimpleNamespace(
        img_w=img_size, img_h=img_size,
        num_classes=2, bg_weight=0.4, ignore_label=255,
        test_parameters=SimpleNamespace(
            conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes,
        ),
        max_lanes=max_lanes, ori_img_h=img_size, ori_img_w=img_size,
        cut_height=0,
    )
    allowed = {
        'test_parameters', 'max_lanes', 'ori_img_h', 'ori_img_w',
        'cut_height', 'cls_loss_weight', 'xyt_loss_weight',
        'iou_loss_weight', 'seg_loss_weight',
    }
    cfg.haskey = _AllowedKeysHasKey(allowed)
    return cfg


class LaneBezierHead(nn.Module):
    """Lane head emitting 16-D cubic-Bezier predictions, optionally with LCM.

    Output dict from forward:
        {
          'lane_output': {
            'predictions_lists': [tensor(B, P, 16)],   # one per refine stage
            'seg': tensor(B, 2, H, W)                  # CLRHead's aux seg head
          },
          'lcm_output': {
            'degree_weights': tensor(B, P, 3),
            'complexity_score': scalar tensor,
            'effective_K': tensor(B, P),
          } | None,
        }
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        ns_classes: int = 1,
        img_size: int = 640,
        num_priors: int = 192,
        num_points: int = 72,
        refine_layers: int = 1,
        sample_points: int = 36,
        fc_hidden_dim: int = 64,
        prior_feat_channels: int = 64,
        max_lanes: int = 8,
        use_square_priors: bool = True,
        use_lcm: bool = False,
        delta_scale: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.num_priors = num_priors
        self.num_points = num_points
        self.refine_layers = refine_layers
        self.max_lanes = max_lanes
        self.use_lcm = use_lcm
        # Scale on the control-point deltas. Without scaling, an
        # unconstrained `nn.Linear` output of magnitude 5 would shoot the
        # control points to (init + 5) ~ [-4, 6] in normalized space.
        # Scale=0.2 keeps the per-stage delta in roughly +/-1 even when
        # the upstream backbone produces high-variance features. Matches
        # appendix sec 5.5 Change 2.
        self.delta_scale = delta_scale

        # P2 adapter (256 -> 64), reused unchanged
        self.adapter = GCAtoCLRAdapter(
            in_channels=hidden_dim, out_channels=prior_feat_channels,
            num_levels=3,
        )

        clr_cfg = _build_clr_cfg(img_size=img_size, max_lanes=max_lanes)
        if use_square_priors:
            head_cls = CLRHeadForSquareImage
        else:
            from clrkd.models.heads.clr_head import CLRHead as _BaseCLRHead
            head_cls = _BaseCLRHead
        self.lane_head = head_cls(
            num_points=num_points,
            prior_feat_channels=prior_feat_channels,
            fc_hidden_dim=fc_hidden_dim,
            num_priors=num_priors,
            num_fc=2,
            refine_layers=refine_layers,
            sample_points=sample_points,
            cfg=clr_cfg,
        )
        # Detach non-leaf priors buffers for ModelEMA's deepcopy (same
        # bug fix as P4 LaneSegHead). The priors get reassigned each
        # forward, so this only matters for the initial deepcopy.
        with torch.no_grad():
            self.lane_head.priors = self.lane_head.priors.detach().clone()
            self.lane_head.priors_on_featmap = (
                self.lane_head.priors_on_featmap.detach().clone()
            )

        # Replace CLRHead's reg_layers (originally 76-D for polyline) with
        # a Bezier-specific 12-D output:
        #   reg[:, :, 0:8]  = control-point deltas (x4, y4)
        #   reg[:, :, 8]    = t_start logit
        #   reg[:, :, 9]    = t_end logit
        #   reg[:, :, 10]   = complexity score logit (heuristic)
        #   reg[:, :, 11]   = reserved
        self.lane_head.reg_layers = nn.Linear(fc_hidden_dim, 12)
        nn.init.normal_(self.lane_head.reg_layers.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lane_head.reg_layers.bias)
        # CRITICAL init: t_start/t_end logits must bias the curve to span
        # (almost) the FULL Bezier at step 0. With a zero bias,
        # t_start=sigmoid(0)=0.5 and t_end=max(0.5, 0.55)=0.55, so each
        # curve renders only the t in [0.50, 0.55] segment - a 5% stub.
        # That stub barely overlaps the GT line, so the IoU/seg loss feeds
        # almost no gradient back to widen the t-range, and the bezier head
        # stays stuck near IoU~0.02 (vs polyline ~0.089). Biasing slot 8
        # (t_start logit) to -4 -> sigmoid(-4)=0.018 and slot 9 (t_end
        # logit) to +4 -> sigmoid(+4)=0.982 makes the curve span t in
        # [0.018, 0.982] = ~96% of the lane from the very first step, so
        # the head learns to SHAPE a full-length curve instead of having
        # to first discover that it should grow one.
        with torch.no_grad():
            self.lane_head.reg_layers.bias[8] = -4.0   # t_start logit
            self.lane_head.reg_layers.bias[9] = 4.0     # t_end logit

        # cls_layers keeps its 2-D output (neg, pos).

        if use_lcm:
            self.lcm = LaneComplexityModule(
                feat_dim=fc_hidden_dim,
                adapter_mid_dim=32, adapter_out_dim=16, gate_mid_dim=8,
            )
        else:
            self.lcm = None

        # Where to stash the last-stage LCM output so MTDETRDecoder can
        # read it without going through the entire forward dict.
        self._cached_lcm_output: Optional[dict] = None

    # ------------------------------------------------------------------
    def forward(
        self,
        x_proj_seg: List[torch.Tensor],
        imgsz: Optional[int] = None,
        batch: Optional[dict] = None,
    ) -> Dict:
        adapted = self.adapter(x_proj_seg)
        if self.training and batch is not None:
            lane_output = self._lane_forward_train(adapted)
        else:
            lane_output = self._lane_forward_eval(adapted)
        return {
            'lane_output': lane_output,
            'lcm_output': self._cached_lcm_output,
        }

    # ------------------------------------------------------------------
    def _lane_forward_train(self, adapted: List[torch.Tensor]) -> Dict:
        head = self.lane_head
        batch_features = list(adapted[len(adapted) - head.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        head.priors, head.priors_on_featmap = head.generate_priors_from_embeddings()
        priors = head.priors.repeat(batch_size, 1, 1)
        priors_on_featmap = head.priors_on_featmap.repeat(batch_size, 1, 1)

        predictions_lists: List[torch.Tensor] = []
        prior_features_stages: List[torch.Tensor] = []
        lcm_output_last_stage: Optional[dict] = None

        for stage in range(head.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            batch_prior_features = head.pool_prior_features(
                batch_features[stage], num_priors, prior_xs,
            )
            prior_features_stages.append(batch_prior_features)
            fc_features = head.roi_gather(
                prior_features_stages, batch_features[stage], stage,
            )
            fc_features = fc_features.view(num_priors, batch_size, -1).reshape(
                batch_size * num_priors, head.fc_hidden_dim,
            )

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in head.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in head.reg_modules:
                reg_features = reg_layer(reg_features)
            cls_logits = head.cls_layers(cls_features)
            reg = head.reg_layers(reg_features)
            cls_logits = cls_logits.reshape(batch_size, -1, 2)
            reg = reg.reshape(batch_size, -1, 12)

            # ------------------------------------------------------------------
            # Build 16-D Bezier predictions from priors + regression deltas.
            # Assembled via torch.cat (NO in-place slice assignment) so
            # autograd doesn't trip over a read-modify-write on the same
            # slot. An earlier version used `pred_bez = new_zeros(...)`
            # then slice-assigned each field; the t_end correction's
            # read-modify-write `pred_bez[:, :, 11] = max(pred_bez[:, :, 11], ...)`
            # bumped the underlying tensor's autograd version while a
            # prior view was already saved for backward, producing
            # "RuntimeError: variable ... is at version 6; expected 4".
            # ------------------------------------------------------------------
            init_cps = priors_to_bezier_init(priors)              # (B, P, 8)
            new_cps = init_cps + reg[:, :, :8] * self.delta_scale  # (B, P, 8)
            t_start = torch.sigmoid(reg[:, :, 8:9])               # (B, P, 1)
            t_end_raw = torch.sigmoid(reg[:, :, 9:10])            # (B, P, 1)
            # t_end > t_start + 0.05. Detach the start side so the
            # constraint doesn't leak gradient into t_start.
            t_end = torch.maximum(t_end_raw, t_start.detach() + 0.05)
            complexity = torch.sigmoid(reg[:, :, 10:11])          # (B, P, 1)

            # LCM degree weights live in slots [13:16]. When LCM is off,
            # those slots are zeros (still produced from priors so the
            # dtype/device match).
            if self.lcm is not None:
                fc_reshaped = fc_features.view(batch_size, num_priors, -1)
                lcm_out = self.lcm(fc_reshaped)
                degree_weights = lcm_out['degree_weights']        # (B, P, 3)
                if stage == head.refine_layers - 1:
                    lcm_output_last_stage = lcm_out
            else:
                degree_weights = priors.new_zeros(batch_size, num_priors, 3)

            pred_bez = torch.cat([
                cls_logits,        # (B, P, 2)
                new_cps,           # (B, P, 8)
                t_start,           # (B, P, 1)
                t_end,             # (B, P, 1)
                complexity,        # (B, P, 1)
                degree_weights,    # (B, P, 3)
            ], dim=-1)              # -> (B, P, 16)

            predictions_lists.append(pred_bez)

            # Iterative refinement: convert the refined Bezier back into
            # CLR-style (start_y, start_x, theta, length) priors so the
            # downstream pool_prior_features (which works on prior LINES)
            # can be reused for next stage.
            if stage != head.refine_layers - 1:
                refined_P0_x = pred_bez[:, :, 2]
                refined_P3_x = pred_bez[:, :, 5]
                refined_P0_y = pred_bez[:, :, 6]
                refined_P3_y = pred_bez[:, :, 9]
                dx = refined_P3_x - refined_P0_x
                dy = refined_P3_y - refined_P0_y
                new_start_y = refined_P0_y
                new_start_x = refined_P0_x
                # theta normalized to [0, 1]: atan2(dx, -dy) / pi
                new_theta = torch.atan2(dx, -dy) / math.pi
                new_length_norm = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-4)
                # Multiply by 71 strips for compatibility with old prior
                # encoding (length in strip-count, not normalized).
                new_length = (new_length_norm * 71.0).clamp(max=71.0)
                priors = priors.detach().clone()
                priors[:, :, 2] = new_start_y
                priors[:, :, 3] = new_start_x
                priors[:, :, 4] = new_theta
                priors[:, :, 5] = new_length
                # Update priors_on_featmap from new priors' x values at
                # sample_x_indexs (mirrors CLRHead's loop).
                priors_on_featmap = priors[..., 6 + head.sample_x_indexs]

        seg_features = torch.cat([
            F.interpolate(
                feature,
                size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                mode='bilinear', align_corners=False,
            )
            for feature in batch_features
        ], dim=1)
        seg = head.seg_decoder(seg_features)

        self._cached_lcm_output = lcm_output_last_stage
        return {
            'predictions_lists': predictions_lists,
            'seg': seg,
        }

    # ------------------------------------------------------------------
    def _lane_forward_eval(self, adapted: List[torch.Tensor]) -> torch.Tensor:
        """Eval-mode: just return the final 16-D predictions (no seg).

        Mirrors the polyline LaneSegHead's eval path which returns a
        single tensor.
        """
        head = self.lane_head
        batch_features = list(adapted[len(adapted) - head.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        head.priors, head.priors_on_featmap = head.generate_priors_from_embeddings()
        priors = head.priors.repeat(batch_size, 1, 1)
        priors_on_featmap = head.priors_on_featmap.repeat(batch_size, 1, 1)

        prior_features_stages: List[torch.Tensor] = []
        pred_bez = None
        lcm_output_last_stage: Optional[dict] = None

        for stage in range(head.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            batch_prior_features = head.pool_prior_features(
                batch_features[stage], num_priors, prior_xs,
            )
            prior_features_stages.append(batch_prior_features)
            fc_features = head.roi_gather(
                prior_features_stages, batch_features[stage], stage,
            )
            fc_features = fc_features.view(num_priors, batch_size, -1).reshape(
                batch_size * num_priors, head.fc_hidden_dim,
            )

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in head.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in head.reg_modules:
                reg_features = reg_layer(reg_features)
            cls_logits = head.cls_layers(cls_features).reshape(batch_size, -1, 2)
            reg = head.reg_layers(reg_features).reshape(batch_size, -1, 12)

            # Same torch.cat assembly as training to avoid in-place
            # slice writes that would trip autograd. Eval-mode doesn't
            # need backward but keeping the patterns identical makes the
            # two paths consistent and avoids subtle drift.
            init_cps = priors_to_bezier_init(priors)
            new_cps = init_cps + reg[:, :, :8] * self.delta_scale
            t_start = torch.sigmoid(reg[:, :, 8:9])
            t_end_raw = torch.sigmoid(reg[:, :, 9:10])
            t_end = torch.maximum(t_end_raw, t_start.detach() + 0.05)
            complexity = torch.sigmoid(reg[:, :, 10:11])

            if self.lcm is not None:
                fc_reshaped = fc_features.view(batch_size, num_priors, -1)
                lcm_out = self.lcm(fc_reshaped)
                degree_weights = lcm_out['degree_weights']
                if stage == head.refine_layers - 1:
                    lcm_output_last_stage = lcm_out
            else:
                degree_weights = priors.new_zeros(batch_size, num_priors, 3)

            pred_bez = torch.cat([
                cls_logits, new_cps, t_start, t_end, complexity, degree_weights,
            ], dim=-1)

            if stage != head.refine_layers - 1:
                refined_P0_x = pred_bez[:, :, 2]
                refined_P3_x = pred_bez[:, :, 5]
                refined_P0_y = pred_bez[:, :, 6]
                refined_P3_y = pred_bez[:, :, 9]
                dx = refined_P3_x - refined_P0_x
                dy = refined_P3_y - refined_P0_y
                priors = priors.detach().clone()
                priors[:, :, 2] = refined_P0_y
                priors[:, :, 3] = refined_P0_x
                priors[:, :, 4] = torch.atan2(dx, -dy) / math.pi
                priors[:, :, 5] = (torch.sqrt(dx*dx + dy*dy) * 71.0).clamp(max=71.0)
                priors_on_featmap = priors[..., 6 + head.sample_x_indexs]

        self._cached_lcm_output = lcm_output_last_stage
        return pred_bez


__all__ = ['LaneBezierHead', 'priors_to_bezier_init']
