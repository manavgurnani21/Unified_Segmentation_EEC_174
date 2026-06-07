"""Phase P4: LaneSegHead - wraps GCAtoCLRAdapter (P2) + CLRHeadForSquareImage (P3).

Replaces RMT-PPAD's TransformerSegmentationDecoder for the lane-only model.
Drivable area is GONE - this head emits ONLY lane outputs.

Per appendix-path3-implementation-prompt.md sec 7.2 with one practical
deviation: we set `refine_layers=1` to match CLRKDNet's own CULane config
default. The appendix wrote `refine_layers=3`; we'll bump it in P8 ablation
if it helps. With 1, the lane head is ~3x cheaper.

This module sits in P4_lane_head_integration/tools/ for phase-isolation; a
thin re-export at vendor/RMT-PPAD/ultralytics/nn/modules/lane_head.py
makes it importable from the vendored ultralytics package.
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


def _import_by_path(name: str, path: Path):
    """Side-effect-free import of a single .py by absolute path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {name} from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def _find_sibling_phase_modules():
    """Locate the P2 adapter + P3 square-head source files starting from this
    file's location. Works both when this module sits under P4_lane_head_
    integration/tools/ and when re-exported from the vendored ultralytics
    package."""
    here = Path(__file__).resolve()
    # Probe upwards for rmt_ppad_migration/.
    cur = here.parent
    migration_root: Optional[Path] = None
    for _ in range(8):
        if cur.name == 'rmt_ppad_migration' and (cur / 'README.md').exists():
            migration_root = cur
            break
        cur = cur.parent

    # Fallback - explicit traversal from REPO_ROOT/yolop_vehicle_lane.
    if migration_root is None:
        repo_root = here
        for _ in range(8):
            if (repo_root / 'stage2' / 'rmt_ppad_migration').exists():
                migration_root = repo_root / 'stage2' / 'rmt_ppad_migration'
                break
            repo_root = repo_root.parent

    if migration_root is None:
        raise FileNotFoundError(
            f'Could not locate rmt_ppad_migration/ starting from {here}'
        )

    p2 = migration_root / 'P2_adapter' / 'tools' / 'gca_to_clr_adapter.py'
    p3 = migration_root / 'P3_square_clrhead' / 'tools' / 'clr_head_square.py'
    if not p2.exists() or not p3.exists():
        raise FileNotFoundError(f'Missing P2 or P3 source under {migration_root}')
    return p2, p3


_p2_path, _p3_path = _find_sibling_phase_modules()
_p2_mod = _import_by_path('p4_p2_adapter', _p2_path)
_p3_mod = _import_by_path('p4_p3_square_head', _p3_path)

GCAtoCLRAdapter = _p2_mod.GCAtoCLRAdapter
CLRHeadForSquareImage = _p3_mod.CLRHeadForSquareImage


class _AllowedKeysHasKey:
    """Picklable replacement for the closure `lambda k, _a=allowed: k in _a`.

    Why: CLRHead's cfg is stored on the model module as `self.cfg`, and
    `cfg.haskey` is set so CLRHead can probe optional fields. The
    original lambda version isn't picklable - its qualname includes
    `<locals>`, which torch.save / pickle refuses. That broke
    `strip_optimizer()`'s final `torch.save(best.pt)` call at the end
    of training. A class with `__call__` defined at module level is
    fully picklable.
    """

    def __init__(self, allowed_keys):
        self._allowed = frozenset(allowed_keys)

    def __call__(self, key):
        return key in self._allowed


def _build_clr_cfg(img_size: int, max_lanes: int) -> SimpleNamespace:
    """Build the SimpleNamespace cfg CLRHead.__init__ expects."""
    cfg = SimpleNamespace(
        img_w=img_size,
        img_h=img_size,
        num_classes=2,         # binary aux (lane vs bg) for CLRHead's internal aux seg head
        bg_weight=0.4,
        ignore_label=255,
        test_parameters=SimpleNamespace(
            conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes,
        ),
        max_lanes=max_lanes,
        ori_img_h=img_size,
        ori_img_w=img_size,
        cut_height=0,
    )
    allowed_keys = {
        'test_parameters', 'max_lanes', 'ori_img_h', 'ori_img_w',
        'cut_height', 'cls_loss_weight', 'xyt_loss_weight',
        'iou_loss_weight', 'seg_loss_weight',
    }
    cfg.haskey = _AllowedKeysHasKey(allowed_keys)
    return cfg


class LaneSegHead(nn.Module):
    """Lane-only segmentation head: GCA features -> 78-D lane predictions.

    Returns a dict (not a tensor) so the caller can pick out either the
    raw lane predictions OR the CLRHead aux seg map. P5 loss + P6
    dataloader + P7 validator all consume the dict directly.

    Forward signature is `forward(x_proj_seg, imgsz=None, batch=None)`
    to be call-compatible with the call site at
    `MTDETRDecoder.forward()`'s line `self.seg_head(x_proj, self.imgsz)`.
    """

    def __init__(self,
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
                 use_square_priors: bool = True):
        """
        use_square_priors: True (default) -> CLRHeadForSquareImage (32/128/32
            prior split, tuned for 640x640 square inputs). False -> base
            CLRHead from CLRKDNet (24/144/24, CULane's 320x800 tuning). Used
            by P8's no_square_priors ablation row.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ns_classes = ns_classes
        self.img_size = img_size
        self.num_priors = num_priors
        self.num_points = num_points
        self.refine_layers = refine_layers
        self.max_lanes = max_lanes
        self.use_square_priors = use_square_priors

        self.adapter = GCAtoCLRAdapter(
            in_channels=hidden_dim, out_channels=prior_feat_channels,
            num_levels=3,
        )

        clr_cfg = _build_clr_cfg(img_size=img_size, max_lanes=max_lanes)
        # P8 ablation: row 3 (no_square_priors) instantiates the BASE CLRHead
        # from CLRKDNet with its CULane prior split (24/144/24) instead of
        # the 32/128/32 square-tuned variant.
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
        # P8 fix (ModelEMA deepcopy): CLRHead.__init__ does
        #   init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        #   self.register_buffer('priors', init_priors)
        #   self.register_buffer('priors_on_featmap', priors_on_featmap)
        # The two returned tensors derive from `self.prior_embeddings.weight`
        # (a learnable Parameter) via arithmetic - they're NON-LEAF tensors
        # with requires_grad=True. PyTorch's deepcopy (used by ModelEMA
        # during _setup_train) refuses non-leaf grad tensors. Detach + clone
        # them so the initial buffers are leaves; forward() reassigns fresh
        # priors each training step, but by then ModelEMA's one-time
        # deepcopy is already done and EMA updates use in-place ops.
        with torch.no_grad():
            self.lane_head.priors = self.lane_head.priors.detach().clone()
            self.lane_head.priors_on_featmap = self.lane_head.priors_on_featmap.detach().clone()

    def forward(self, x_proj_seg: List[torch.Tensor],
                imgsz: Optional[int] = None,
                batch: Optional[dict] = None) -> Dict:
        """Project channels then run CLRHead.

        Args:
            x_proj_seg: 3 tensors from GCA, each (B, 256, H_i, W_i).
            imgsz:      unused for lane head, kept for call-site compatibility.
            batch:      gt dict during training; passed through.

        Returns:
            dict {'lane_output': ...}
              - training: nested dict {'predictions_lists': [...], 'seg': tensor}
              - eval:     tensor (B, num_priors, 78) - final-stage predictions
        """
        adapted = self.adapter(x_proj_seg)
        if self.training and batch is not None:
            lane_output = self._lane_forward_train(adapted)
        else:
            lane_output = self.lane_head(adapted)
        return {'lane_output': lane_output}

    def _lane_forward_train(self, adapted: List[torch.Tensor]) -> Dict:
        """Run CLRHead's refinement loop WITHOUT triggering its built-in
        loss. We compute the loss externally in P5 via MTDETRDLoss."""
        head = self.lane_head
        batch_features = list(adapted[len(adapted) - head.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        head.priors, head.priors_on_featmap = head.generate_priors_from_embeddings()
        priors = head.priors.repeat(batch_size, 1, 1)
        priors_on_featmap = head.priors_on_featmap.repeat(batch_size, 1, 1)

        predictions_lists = []
        prior_features_stages: List[torch.Tensor] = []
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
            cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits
            predictions[:, :, 2:5] += reg[:, :, :3]
            predictions[:, :, 5] = reg[:, :, 3]

            def _tran(t):
                return t.unsqueeze(2).clone().repeat(1, 1, head.n_offsets)

            predictions[..., 6:] = (
                _tran(predictions[..., 3]) * (head.img_w - 1)
                + ((1 - head.prior_ys.repeat(batch_size, num_priors, 1)
                    - _tran(predictions[..., 2])) * head.img_h
                   / torch.tan(_tran(predictions[..., 4]) * math.pi + 1e-5))
            ) / (head.img_w - 1)
            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != head.refine_layers - 1:
                priors = prediction_lines.detach().clone()
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

        return {'predictions_lists': predictions_lists, 'seg': seg}


__all__ = ['LaneSegHead']
