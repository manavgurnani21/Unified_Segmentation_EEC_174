import argparse
import os, sys, glob
# Auto-fix LD_LIBRARY_PATH for ONNXRuntime CUDAExecutionProvider
nvidia_libs = glob.glob(os.path.join(os.path.dirname(sys.executable), '../lib/python*/site-packages/nvidia/*/lib'))
if nvidia_libs and nvidia_libs[0] not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_libs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torchvision.transforms as transforms
import numpy as np
import onnxruntime as ort
import time

from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xyxy2xywh, xywh2xyxy, box_iou, coco80_to_coco91_class, ap_per_class
from lib.utils.utils import create_logger
from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network with ONNXRuntime')
    parser.add_argument('--onnx', type=str, default='YOLOPX/weights/yolopx_qat_epoch_0_sim.onnx')
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--modelDir', type=str, default='')
    parser.add_argument('--logDir', type=str, default='runs/')
    args = parser.parse_args()
    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def main():
    args = parse_args()
    update_config(cfg, args)
    
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.LOG_DIR, 'test')
    logger.info(pprint.pformat(args))

    device = torch.device('cpu') # For post-processing
    
    print("=== Loading ONNX model in ONNXRuntime (CUDA) ===")
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(args.onnx, providers=providers)
    input_name = session.get_inputs()[0].name

    print("begin to load data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn,
        drop_last=True  # Ensure static batch size for ONNX
    )
    print('load data finished')

    max_stride = 32
    _, imgsz = [check_img_size(x, s=max_stride) for x in cfg.MODEL.IMAGE_SIZE]

    nc = 1
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    confusion_matrix = ConfusionMatrix(nc=1)
    da_metric = SegmentationMetric(cfg.num_seg_class)
    ll_metric = SegmentationMetric(2)

    names = {0: 'vehicle'}
    coco91class = coco80_to_coco91_class()
    
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    T_inf = AverageMeter()
    T_nms = AverageMeter()

    jdict, stats, ap, ap_class = [], [], [], []
    seen = 0

    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        img_np = img.numpy()
        b, c, h, w = img_np.shape
        
        # Pad to 640x640 for static ONNX graph
        if h != 640 or w != 640:
            pad_h_extra = 640 - h
            pad_w_extra = 640 - w
            img_np = np.pad(img_np, ((0, 0), (0, 0), (0, pad_h_extra), (0, pad_w_extra)), mode='constant')

        # INFERENCE
        t0 = time.time()
        outs = session.run(None, {input_name: img_np})
        t_inf = time.time() - t0
        if batch_i > 0:
            T_inf.update(t_inf / img.size(0), img.size(0))

        # Unpack ONNX outputs (flattened)
        inf_out_np = outs[0]
        da_seg_out_np = outs[4]
        ll_seg_out_np = outs[5]
        
        # Slice segmentation outputs back to the unpadded dynamic shape
        da_seg_out_np = da_seg_out_np[:, :, :h, :w]
        ll_seg_out_np = ll_seg_out_np[:, :, :h, :w]

        # Convert back to PyTorch for post-processing
        inf_out = torch.from_numpy(inf_out_np).to(device)
        da_seg_out = torch.from_numpy(da_seg_out_np).to(device)
        ll_seg_out = torch.from_numpy(ll_seg_out_np).to(device)

        nb, _, height, width = img.shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w, pad_h = int(pad_w), int(pad_h)

        # DA Evaluation
        _, da_predict = torch.max(da_seg_out, 1)
        _, da_gt = torch.max(target[1], 1)
        da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        da_metric.reset()
        da_metric.addBatch(da_predict, da_gt)
        da_acc_seg.update(da_metric.pixelAccuracy(), img.size(0))
        da_IoU_seg.update(da_metric.IntersectionOverUnion(), img.size(0))
        da_mIoU_seg.update(da_metric.meanIntersectionOverUnion(), img.size(0))

        # LL Evaluation
        _, ll_predict = torch.max(ll_seg_out, 1)
        _, ll_gt = torch.max(target[2], 1)
        ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        ll_metric.reset()
        ll_metric.addBatch(ll_predict, ll_gt)
        ll_acc_seg.update(ll_metric.lineAccuracy(), img.size(0))
        ll_IoU_seg.update(ll_metric.IntersectionOverUnion(), img.size(0))
        ll_mIoU_seg.update(ll_metric.meanIntersectionOverUnion(), img.size(0))

        # NMS
        t1 = time.time()
        lb = []
        output = non_max_suppression(inf_out, conf_thres=args.conf_thres, iou_thres=args.iou_thres, labels=lb)
        t_nms = time.time() - t1
        if batch_i > 0:
            T_nms.update(t_nms / img.size(0), img.size(0))

        nlabel = (target[0].sum(dim=2) > 0).sum(dim=1)
        for si, pred in enumerate(output):
            nl = int(nlabel[si])
            labels = target[0][si, :nl, 0:5]
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                
                for cls_idx in torch.unique(tcls_tensor):                    
                    ti = (cls_idx == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls_idx == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            stats.append((correct, pred[:, 4], pred[:, 5], tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir='', names=names)
        ap50, ap70, ap75, ap = ap[:, 0], ap[:,4], ap[:,5], ap.mean(1)
        mp, mr, map50, map70, map75, map_val = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        mp, mr, map50, map_val = 0.0, 0.0, 0.0, 0.0
        nt = torch.zeros(1)

    msg = (f'Test Complete:\n'
           f'Driving area Segment: Acc({da_acc_seg.avg:.3f})    IOU ({da_IoU_seg.avg:.3f})    mIOU({da_mIoU_seg.avg:.3f})\n'
           f'Lane line Segment: Acc({ll_acc_seg.avg:.3f})    IOU ({ll_IoU_seg.avg:.3f})  mIOU({ll_mIoU_seg.avg:.3f})\n'
           f'Detect: P({mp:.3f})  R({mr:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map_val:.3f})\n'
           f'Time: inference({T_inf.avg:.4f}s/frame)  nms({T_nms.avg:.4f}s/frame)')
    logger.info(msg)
    print(msg)

if __name__ == '__main__':
    main()