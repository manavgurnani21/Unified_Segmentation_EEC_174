from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.fusion.detection import DetectionLossConfig, SimpleVehicleDetectionHead, SimpleVehicleDetectionLoss, read_yolo_label
from stage2.fusion.lane_head import CurveLaneHead
from stage2.fusion.lane_targets import soft_polyline_mask_numpy
from stage2.fusion.losses import FusionLaneLoss, FusionLossConfig, UncertaintyMultiTaskLoss, compute_grad_cosine
from stage2.fusion.model import FusionModel


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, channels // 2)
        self.net = nn.Sequential(ConvBNAct(channels, hidden, 1, 1), ConvBNAct(hidden, channels, 3, 1))

    def forward(self, x):
        return x + self.net(x)


class TinyCSPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 64, 3, 2)
        self.s2 = nn.Sequential(ConvBNAct(64, 128, 3, 2), ResidualBlock(128), ResidualBlock(128))
        self.s3 = nn.Sequential(ConvBNAct(128, 256, 3, 2), ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.s4 = nn.Sequential(ConvBNAct(256, 512, 3, 2), ResidualBlock(512), ResidualBlock(512), ResidualBlock(512))

    def forward(self, x):
        x = self.stem(x)
        p3 = self.s2(x)
        p4 = self.s3(p3)
        p5 = self.s4(p4)
        return [p3, p4, p5]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_tar_once(tar_path: Path, dest: Path) -> None:
    marker = dest / '.extract.ok'
    if marker.exists():
        print(f'Already extracted: {tar_path} -> {dest}')
        return
    if not tar_path.exists():
        raise FileNotFoundError(f'Missing tar archive: {tar_path}')
    if dest.exists():
        shutil.rmtree(dest)
    ensure_dir(dest)
    subprocess.check_call(['tar', '-xf', str(tar_path), '-C', str(dest)])
    marker.write_text(str(tar_path), encoding='utf-8')


def resample_polyline(points: np.ndarray, n: int, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return np.zeros((n, 2), dtype=np.float32), np.zeros((n,), dtype=np.float32)
    diffs = points[1:] - points[:-1]
    seg_len = np.sqrt((diffs * diffs).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])
    if total <= 1e-6:
        return np.zeros((n, 2), dtype=np.float32), np.zeros((n,), dtype=np.float32)
    target = np.linspace(0.0, total, n)
    xs = np.interp(target, arc, points[:, 0]) / float(img_w)
    ys = np.interp(target, arc, points[:, 1]) / float(img_h)
    out = np.stack([xs, ys], axis=1).astype(np.float32)
    vis = ((out[:, 0] >= 0) & (out[:, 0] <= 1) & (out[:, 1] >= 0) & (out[:, 1] <= 1)).astype(np.float32)
    return np.clip(out, 0.0, 1.0), vis


def read_lines_targets(path: Path, max_lanes: int, num_points: int, img_w: int, img_h: int) -> Dict[str, np.ndarray]:
    existence = np.zeros((max_lanes,), dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.full((max_lanes,), -1, dtype=np.int64)
    rows = []
    if path.exists():
        rows = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    parsed = []
    for row in rows:
        vals = [float(x) for x in row.split()]
        if len(vals) < 4 or len(vals) % 2 != 0:
            continue
        arr = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        yspan = float(arr[:, 1].max() - arr[:, 1].min())
        length = float(np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=1)).sum())
        if length >= 2.0 and yspan >= 1.0:
            parsed.append((yspan, length, arr))
    parsed.sort(key=lambda x: (-x[0], -x[1]))
    for i, (_ys, _ln, arr) in enumerate(parsed[:max_lanes]):
        pts, vis = resample_polyline(arr, num_points, img_w, img_h)
        existence[i] = 1.0
        points[i] = pts
        visibility[i] = vis
        lane_type[i] = 0
    return {'existence': existence, 'points': points, 'visibility': visibility, 'lane_type': lane_type}


class BDDCurveFusionDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: Tuple[int, int], aux_mask_size: Tuple[int, int], max_lanes: int, num_points: int, det_label_root: Optional[Path] = None, limit: int = 0):
        self.root = Path(root)
        self.split = split
        self.image_size = tuple(image_size)
        self.aux_mask_size = tuple(aux_mask_size)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.det_label_root = det_label_root
        list_file = self.root / 'list' / ('train_gt.txt' if split == 'train' else 'val.txt')
        if not list_file.exists():
            raise FileNotFoundError(f'Missing split list: {list_file}')
        items = []
        for row in list_file.read_text(encoding='utf-8').splitlines():
            if not row.strip():
                continue
            rel = row.split()[0].lstrip('/')
            items.append(self.root / rel)
        if limit and limit > 0:
            items = items[:limit]
        self.items = items
        if not self.items:
            raise RuntimeError(f'No samples found in {list_file}')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path = self.items[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        org_h, org_w = image.shape[:2]
        resized = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        line_path = img_path.with_suffix('.lines.txt')
        lane = read_lines_targets(line_path, self.max_lanes, self.num_points, org_w, org_h)
        mask = soft_polyline_mask_numpy(lane['points'], lane['visibility'], height=self.aux_mask_size[0], width=self.aux_mask_size[1])
        target = {k: torch.from_numpy(v) for k, v in lane.items()}
        target['mask_target'] = torch.from_numpy(mask[None, ...].astype(np.float32))
        det_labels = torch.zeros((0, 5), dtype=torch.float32)
        if self.det_label_root is not None:
            label_path = self.det_label_root / self.split / f'{img_path.stem}.txt'
            det_labels = read_yolo_label(label_path)
        return tensor, target, det_labels, img_path.name


def collate_fn(batch):
    images, targets, det_labels, names = zip(*batch)
    images = torch.stack(images, 0)
    merged = {}
    for key in targets[0].keys():
        merged[key] = torch.stack([t[key] for t in targets], 0)
    return images, merged, list(det_labels), list(names)


def tensor_dict_to_device(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in data.items()}


def build_model(cfg: Dict) -> FusionModel:
    backbone = TinyCSPBackbone()
    det_head = SimpleVehicleDetectionHead(in_channels=128, hidden_dim=128, num_classes=int(cfg['model']['detection_head'].get('num_classes', 1)))
    lane_cfg = cfg['model']['lane_head']
    lane_head = CurveLaneHead(
        in_channels=cfg['model']['feature_channels'],
        embed_dim=int(lane_cfg.get('embed_dim', 128)),
        max_lanes=int(lane_cfg.get('max_lanes', 10)),
        num_points=int(lane_cfg.get('num_points', 72)),
        mask_size=tuple(lane_cfg.get('mask_size', [72, 128])),
        mask_aux=bool(lane_cfg.get('mask_aux', True)),
        num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
    )
    return FusionModel(backbone=backbone, feature_channels=cfg['model']['feature_channels'], detection_head=det_head, lane_head=lane_head)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train Stage 2 basic detection + CLRKD curve-lane fusion.')
    parser.add_argument('--config', default='/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/configs/rmt_clrkd_basic_fusion.yaml')
    parser.add_argument('--curve-tar', default=None)
    parser.add_argument('--curve-root', default=None)
    parser.add_argument('--det-label-root', default=None)
    parser.add_argument('--work-dir', default='/content/stage2_rmt_clrkd_basic_fusion')
    parser.add_argument('--output-tar', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--no-grad-cosine', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    curve_tar = Path(args.curve_tar or cfg['dataset']['bdd_archive'])
    curve_root = Path(args.curve_root or cfg['dataset']['local_dir'])
    output_tar = Path(args.output_tar or cfg['run']['output_tar'])
    work_dir = ensure_dir(Path(args.work_dir))
    extract_tar_once(curve_tar, curve_root)

    det_label_root = Path(args.det_label_root) if args.det_label_root else None
    if det_label_root is not None and not det_label_root.exists():
        raise FileNotFoundError(f'Detection label root does not exist: {det_label_root}')
    if det_label_root is None:
        print('Detection label root was not provided. Detection loss will run with empty targets, so this is not a full detection experiment.')
    else:
        print(f'Using detection labels from {det_label_root}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = tuple(cfg['dataset']['image_size'])
    aux_mask_size = tuple(cfg['dataset']['aux_mask_size'])
    train_set = BDDCurveFusionDataset(
        curve_root,
        'train',
        image_size=image_size,
        aux_mask_size=aux_mask_size,
        max_lanes=int(cfg['dataset']['max_lanes']),
        num_points=int(cfg['dataset']['num_points']),
        det_label_root=det_label_root,
        limit=args.limit_train,
    )
    loader = DataLoader(train_set, batch_size=args.batch_size or int(cfg['train']['batch_size']), shuffle=True, num_workers=int(cfg['train'].get('workers', 2)), pin_memory=True, collate_fn=collate_fn)

    model = build_model(cfg).to(device)
    lane_cfg = cfg['loss']['lane']
    lane_loss = FusionLaneLoss(FusionLossConfig(lambda_lane=float(cfg['loss'].get('lambda_lane', 1.0)), use_uncertainty=bool(cfg['loss'].get('use_uncertainty', False)), **lane_cfg)).to(device)
    det_cfg = DetectionLossConfig(**cfg['loss']['det'])
    det_loss = SimpleVehicleDetectionLoss(det_cfg).to(device)
    mtl = UncertaintyMultiTaskLoss(2).to(device) if bool(cfg['loss'].get('use_uncertainty', False)) else None
    params = list(model.parameters()) + list(lane_loss.parameters())
    if mtl is not None:
        params += list(mtl.parameters())
    opt = optim.AdamW(params, lr=float(cfg['train']['lr0']), weight_decay=float(cfg['train']['weight_decay']))
    epochs = args.epochs or int(cfg['train']['end_epoch'])
    metrics: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        sums: Dict[str, float] = {}
        count = 0
        for step, (images, lane_targets, det_targets, _names) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            lane_targets = tensor_dict_to_device(lane_targets, device)
            out = model(images)
            l_lane, lane_comp = lane_loss(out['lane'], lane_targets)
            l_det, det_comp = det_loss(out['det'], det_targets)
            if mtl is None:
                total = l_det + float(cfg['loss'].get('lambda_lane', 1.0)) * l_lane
                mtl_comp = {}
            else:
                total, mtl_comp = mtl([l_det, l_lane])
            grad_cos = float('nan')
            if step == 1 and not args.no_grad_cosine and bool(cfg.get('eval', {}).get('log_grad_cosine', True)):
                try:
                    grad_cos = compute_grad_cosine(l_det, l_lane, model.backbone.parameters())
                except RuntimeError:
                    grad_cos = float('nan')
            opt.zero_grad(set_to_none=True)
            total.backward()
            clip = float(cfg['train'].get('grad_clip_norm', 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            comp = {'total': total.detach(), 'grad_cosine': torch.tensor(grad_cos)}
            comp.update(lane_comp)
            comp.update(det_comp)
            comp.update(mtl_comp)
            for key, value in comp.items():
                val = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
                if not np.isnan(val):
                    sums[key] = sums.get(key, 0.0) + val
            count += 1
            if step % 50 == 0:
                print(f'epoch={epoch} step={step} total={float(total.detach().cpu()):.4f} det={float(l_det.detach().cpu()):.4f} lane={float(l_lane.detach().cpu()):.4f}')
        epoch_metrics = {k: v / max(1, count) for k, v in sums.items()}
        epoch_metrics['epoch'] = float(epoch)
        metrics.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))
        ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'config': cfg}
        torch.save(ckpt, work_dir / 'last.pt')
        if epoch % int(cfg['train'].get('save_every', 5)) == 0:
            torch.save(ckpt, work_dir / f'epoch_{epoch}.pt')
        (work_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        (work_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    ensure_dir(output_tar.parent)
    if output_tar.exists():
        output_tar.unlink()
    subprocess.check_call(['tar', '-cf', str(output_tar), '-C', str(work_dir), '.'])
    print(f'Wrote {output_tar}')


if __name__ == '__main__':
    main()
