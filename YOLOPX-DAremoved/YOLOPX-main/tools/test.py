# FILE: tools/test.py

import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np

from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter

import lib.dataset as dataset

from lib.config import cfg
from lib.config import update_config

from lib.core.loss import get_loss
from lib.core.function import validate
from lib.core.general import fitness

from lib.models import get_net

from lib.utils.utils import create_logger, select_device


def parse_args():

    parser = argparse.ArgumentParser(
        description='Test Multitask network'
    )

    parser.add_argument(
        '--modelDir',
        help='model directory',
        type=str,
        default=''
    )

    parser.add_argument(
        '--logDir',
        help='log directory',
        type=str,
        default='runs/'
    )

    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default='weights/epoch-195.pth',
        help='model.pth path(s)'
    )

    parser.add_argument(
        '--conf_thres',
        type=float,
        default=0.001,
        help='object confidence threshold'
    )

    parser.add_argument(
        '--iou_thres',
        type=float,
        default=0.6,
        help='IOU threshold for NMS'
    )

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg,
        cfg.LOG_DIR,
        'test'
    )

    logger.info(pprint.pformat(args))

    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print("begin to bulid up model...")

    device = select_device(
        logger,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    ) if not cfg.DEBUG else select_device(logger, 'cpu')

    model = get_net(cfg)

    print("finish build model")

    criterion = get_loss(cfg, device, model)

    model_dict = model.state_dict()

    checkpoint_file = args.weights[0] if isinstance(args.weights, list) else args.weights

    logger.info(
        "=> loading checkpoint '{}'".format(checkpoint_file)
    )

    checkpoint = torch.load(checkpoint_file)

    checkpoint_dict = checkpoint['state_dict']

    model_dict.update(checkpoint_dict)

    model.load_state_dict(model_dict)

    logger.info(
        "=> loaded checkpoint '{}' ".format(checkpoint_file)
    )

    model = model.to(device)

    model.gr = 1.0

    model.nc = 1

    print('bulid model finished')

    print("begin to load data")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval(
        'dataset.' + cfg.DATASET.DATASET
    )(
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
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    print('load data finished')

    epoch = 0

    ll_segment_results, detect_results, total_loss, maps, times = validate(
        epoch,
        cfg,
        valid_loader,
        valid_dataset,
        model,
        criterion,
        final_output_dir,
        tb_log_dir,
        writer_dict,
        logger,
        device
    )

    fi = fitness(
        np.array(detect_results).reshape(1, -1)
    )

    msg = (
        'Test:    Loss({loss:.3f})\n'
        'Lane line Segment: '
        'Acc({ll_seg_acc:.3f})    '
        'IOU ({ll_seg_iou:.3f})    '
        'mIOU({ll_seg_miou:.3f})\n'
        'Detect: '
        'P({p:.3f})  '
        'R({r:.3f})  '
        'mAP@0.5({map50:.3f})  '
        'mAP@0.5:0.95({map:.3f})\n'
        'Time: '
        'inference({t_inf:.4f}s/frame)  '
        'nms({t_nms:.4f}s/frame)'
    ).format(
        loss=total_loss,
        ll_seg_acc=ll_segment_results[0],
        ll_seg_iou=ll_segment_results[1],
        ll_seg_miou=ll_segment_results[2],
        p=detect_results[0],
        r=detect_results[1],
        map50=detect_results[2],
        map=detect_results[3],
        t_inf=times[0],
        t_nms=times[1]
    )

    logger.info(msg)

    print("test finish")


if __name__ == '__main__':

    main()