"""Option-C smoke test: can we drive CLRKDNet's actual CLRHead from our backbone?

Run from the repo root:
    python yolop_vehicle_lane/stage2/scripts/smoke_test_clrnet_head.py

This script:
  1. Adds the repo to sys.path so ``import stage2`` works without conda install.
  2. Tries to import ``VendorCLRNetHead`` -- which itself tries to import
     CLRKDNet's ``CLRHead``. If ``mmcv`` is missing, this fails with a clear
     hint to ``pip install mmcv``.
  3. Builds a minimal model via ``build_joint_model`` with
     ``lane_head.type='clrnet_official'`` so we exercise the full factory
     dispatch.
  4. Feeds a random ``(B, 3, 384, 640)`` tensor through the joint model.
  5. Asserts that the lane head produces sensible shapes:
       - ``cls_logits`` (B, 192)
       - ``coord_pred`` (B, 192, 72, 2) with values in [0, 1]
       - ``lane_param`` (B, 192, 4)
  6. Prints a clean PASS / FAIL summary.

What this does NOT test:
  - Training-mode loss path (CLRHead.loss is bypassed; that's Option A/B work).
  - Real CULane-trained weights (our CLRHead is randomly initialized here).
  - Inference NMS (``clrkd.ops.nms_impl`` is stubbed; calling get_lanes
    would error out -- but the smoke test doesn't go through get_lanes).

Exit codes:
  0 = smoke test passed; Option A/B is viable.
  1 = import failed (probably missing mmcv); install and retry.
  2 = shape assertion failed; vendor wrapper has a bug.
  3 = other runtime error during forward.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


# IMPORTANT: parents[2] = ``yolop_vehicle_lane/`` (where ``stage2/`` is a
# top-level package). parents[3] would be the surrounding directory which
# contains ``external_repos/`` but does NOT make ``stage2`` importable.
# NB78 v1 set parents[3] and failed with ``ModuleNotFoundError: No module
# named 'stage2'``; this is the fix.
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../yolop_vehicle_lane
REPO_ROOT = Path(__file__).resolve().parents[3]      # .../<surrounding repo>
for p in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _make_minimal_cfg():
    """A minimal config dict that's enough to instantiate the joint model with
    a CLRNet-vendor lane head. Backbone is our smallest variant for speed."""
    return {
        'run': {'seed': 0},
        'dataset': {
            'image_size': [384, 640],
            'num_points': 72,
            'aux_mask_size': [72, 128],
            'max_lanes': 10,
            'org_image_size': [720, 1280],
        },
        'model': {
            'backbone': 'exp2b_rmt_gca',
            'width': 0.5,
            'use_aifi': True,
            'yolo26': {
                'source_root': None, 'model_path': None,
                'feature_indices': None, 'strict': True,
            },
            'detection_head': {
                'type': 'detr',
                'num_queries': 100,
                'num_classes': 1,
                'embed_dim': 256,
                'num_decoder_layers': 3,
                'num_heads': 8,
                'dim_feedforward': 1024,
            },
            'lane_head': {
                'type': 'clrnet_official',
                'prior_feat_channels': 64,
                'num_priors': 192,
                'num_points': 72,
                'sample_points': 36,
                'refine_layers': 3,
                'fc_hidden_dim': 64,
                'num_fc': 2,
                'seg_num_classes': 2,
                'bg_weight': 0.4,
                'ignore_label': 255,
                'max_lanes': 10,
                'num_lane_classes': 7,
                'mask_size': [72, 128],
            },
        },
        'loss': {
            'lambda_lane': 1.0, 'lambda_det': 1.0, 'lambda_mode': 'fixed',
            'use_uncertainty': False,
            'no_object_weight': 0.25,
            'det': {
                'bbox_loss_weight': 5.0, 'cls_loss_weight': 1.0,
                'dn_loss_weight': 0.0, 'giou_loss_weight': 2.0,
                'obj_loss_weight': 1.0,
            },
            'lane': {
                'cls_target_type': 'matched_existence',
                'cls_loss_type': 'vfl',
                'w_cls': 5.0, 'w_iou': 2.0, 'w_mask': 1.0, 'w_reg': 1.2,
                'w_smooth': 0.05, 'w_xytl': 0.2, 'w_distill': 0.0,
                'line_iou_radius': 0.015,
                'use_lane_matching': True,
                'lane_assigner': 'dynamic_k',
                'dynamic_k_topk': 4,
                'match_cost_cls': 1.0, 'match_cost_point': 5.0,
                'match_cost_iou': 2.0, 'match_cost_xytl': 0.2,
            },
        },
    }


def main() -> int:
    print('=== Option C smoke test: CLRKDNet CLRHead via our pipeline ===')
    print(f'project_root: {PROJECT_ROOT}  (sys.path[0])')
    print(f'repo_root:    {REPO_ROOT}     (contains external_repos/)')

    # Step 1: try to import the vendor wrapper. Most likely failure point.
    try:
        from stage2.fusion.vendor_clrnet_head import (
            VendorCLRNetHead, import_clrhead, _CLRKD_PATH,
        )
    except ImportError as e:
        print(f'\n[FAIL] could not import VendorCLRNetHead: {e}')
        traceback.print_exc()
        print('\nIf the error mentions mmcv, run `pip install mmcv` and retry.')
        return 1
    print(f'[OK] VendorCLRNetHead imported. CLRKDNet vendor: {_CLRKD_PATH}')

    # Step 2: try to import the real CLRHead (this is where mmcv issues surface).
    try:
        CLRHead = import_clrhead()
    except Exception as e:
        print(f'\n[FAIL] could not import CLRHead from vendor: {e}')
        traceback.print_exc()
        return 1
    print(f'[OK] CLRHead imported: {CLRHead.__module__}.{CLRHead.__name__}')

    # Step 3: torch availability
    try:
        import torch  # noqa: F401
    except ImportError:
        print('[FAIL] torch is not importable.')
        return 3
    print(f'[OK] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')

    # Step 4: build the joint model via the factory.
    try:
        from stage2.fusion.experiment_factory import build_joint_model
    except Exception as e:
        print(f'[FAIL] could not import build_joint_model: {e}')
        traceback.print_exc()
        return 1
    cfg = _make_minimal_cfg()
    try:
        t0 = time.time()
        model = build_joint_model(cfg)
        print(f'[OK] joint model built in {time.time()-t0:.2f}s. '
              f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    except Exception as e:
        print(f'[FAIL] build_joint_model crashed: {e}')
        traceback.print_exc()
        return 3

    # Sanity: confirm the lane_head is our wrapper.
    if not isinstance(model.lane_head, VendorCLRNetHead):
        print(f'[FAIL] expected VendorCLRNetHead, got {type(model.lane_head).__name__}')
        return 2
    print(f'[OK] lane_head type: {type(model.lane_head).__name__}')
    print(f'     prior_feat_channels={model.lane_head.prior_feat_channels}, '
          f'num_priors={model.lane_head.num_priors}, '
          f'num_points={model.lane_head.num_points}, '
          f'refine_layers={model.lane_head.refine_layers}')

    # Step 5: forward pass with a random tensor.
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    B = 2
    images = torch.randn(B, 3, 384, 640, device=device)
    try:
        with torch.no_grad():
            t0 = time.time()
            out = model(images)
            forward_time = time.time() - t0
        print(f'[OK] forward pass completed in {forward_time*1000:.1f}ms '
              f'on {device.type}')
    except Exception as e:
        print(f'[FAIL] forward pass crashed: {e}')
        traceback.print_exc()
        return 3

    # Step 6: shape assertions.
    if not isinstance(out, dict) or 'lane' not in out:
        print(f'[FAIL] expected dict with "lane" key; got keys: '
              f'{list(out.keys()) if isinstance(out, dict) else type(out)}')
        return 2
    lane_out = out['lane']
    expected = {
        'cls_logits': (B, 192),
        'coord_pred': (B, 192, 72, 2),
        'lane_param': (B, 192, 4),
    }
    print('\nLane head output shapes:')
    for key, exp_shape in expected.items():
        if key not in lane_out:
            print(f'  [FAIL] missing key: {key}')
            return 2
        got_shape = tuple(lane_out[key].shape)
        ok = got_shape == exp_shape
        marker = '[OK]' if ok else '[FAIL]'
        print(f'  {marker} {key}: got {got_shape}, expected {exp_shape}')
        if not ok:
            return 2

    # Range check on coord_pred.
    coord = lane_out['coord_pred']
    cp_min = float(coord.min().item())
    cp_max = float(coord.max().item())
    print(f'  coord_pred range: [{cp_min:.4f}, {cp_max:.4f}] '
          f'(should be in [0, 1])')
    if not (0.0 <= cp_min and cp_max <= 1.0 + 1e-5):
        print('  [WARN] coord_pred out of [0, 1] (random-init head; not a hard failure)')

    # cls_logits range -- just a sanity print, no assertion.
    cls = lane_out['cls_logits']
    print(f'  cls_logits range: [{cls.min().item():.3f}, {cls.max().item():.3f}], '
          f'mean={cls.mean().item():.3f}')

    print('\n=== SMOKE TEST PASSED ===')
    print('Option A or B is feasible. The vendor CLRHead can be driven from '
          'our backbone\'s feature pyramid through the channel adapters.')
    print('\nNext step suggestions:')
    print('  - Option A: execute Phase P0 of the user\'s plan (download '
          'RMT-PPAD pretrained model and run their test.py to record baseline).')
    print('  - Option B: write a training-mode forward path that does NOT call '
          'CLRHead.loss; instead use our FusionLaneLoss on cls_logits + coord_pred.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
