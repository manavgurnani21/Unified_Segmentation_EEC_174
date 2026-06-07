# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import YOLODataset, converter
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics, ConfusionMatrix, SegmentationMetric, AverageMeter
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import output_to_target, plot_images
import os
from pathlib import Path
import numpy as np
import torch.nn.functional as F

__all__ = ("MTDETRValidator",)  # tuple or list


class MTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, **kwargs)

    # NOTE: add stretch version load_image for RTDETR mosaic
    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Temporary, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                merge_mask=True,
                imgsz=self.imgsz,
                seg_nc=len(self.data['type_task']['segmentation']),
            )
        )
        return transforms


class MTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Example:
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        validator = RTDETRValidator(args=args)
        validator()
        ```

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def __init__(self, dataloader=None, number_task=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.number_task = number_task
        self.plot_masks = None
        self.process = None
        self.args.task = "multi"
        ### JW init two metics for different task.
        self.metrics = {'detection': DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot), 'segmentation': SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)}
        if not isinstance(self.args.mask_threshold, list):
            self.args.mask_threshold = [self.args.mask_threshold]
        self.mask_thr = torch.tensor(self.args.mask_threshold).view(1, len(self.args.mask_threshold), 1, 1)   ### JW set the mask_threshold

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = self.number_task
        self.filtered_names = {task: {i: self.names[i] for i in indices if i in self.names} for task, indices in self.data['type_task'].items()}
        self.metrics['detection'].names = self.filtered_names['detection']
        self.metrics['detection'].plot = self.args.plots
        self.metrics['segmentation'].names = self.filtered_names['segmentation']
        self.metrics['segmentation'].plot = self.args.plots
        ### JW init the segmentation task
        self.seg_metrics = {self.dataloader.dataset.data['names'][i]: SegmentationMetric(2) for i in self.dataloader.dataset.data['type_task']['segmentation']}
        self.seg_result = {self.dataloader.dataset.data['names'][i]: {'pixacc': AverageMeter(), 'subacc': AverageMeter(), 'IoU': AverageMeter(),
                                  'mIoU': AverageMeter()} for i in self.dataloader.dataset.data['type_task']['segmentation']}
        self.plot_masks = {self.dataloader.dataset.data['names'][i]: [] for i in self.dataloader.dataset.data['type_task']['segmentation']}
        self.confusion_matrix_detection = ConfusionMatrix(nc=self.nc['detection'], conf=self.args.conf)
        self.confusion_matrix_segmentation = ConfusionMatrix(nc=self.nc['segmentation'], conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        # batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return MTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        # P7: branch on the seg output shape. LaneSegHead returns a dict
        # `{'lane_output': <list-of-stages | (B, num_priors, 78) tensor>}`;
        # the legacy seg head returns a (B, n_seg, H, W) sigmoid mask.
        seg_part = preds[1][5]
        if isinstance(seg_part, dict):
            from ultralytics.models.utils.lane_rasterize import (
                batch_lanes_to_mask, lane_score_stats,
            )
            lane_output = seg_part.get('lane_output')
            # During training the forward returns the multi-stage dict
            # {'predictions_lists': [...], 'seg': ...}; during eval it's a
            # raw (B, num_priors, 78) tensor (final-stage predictions).
            if isinstance(lane_output, dict):
                final_preds = lane_output['predictions_lists'][-1]
            elif isinstance(lane_output, (list, tuple)):
                final_preds = lane_output[-1]
            else:
                final_preds = lane_output

            # Phase 1: accumulate per-prior pos-score stats so get_stats()
            # can print a histogram and expose a score-spread metric (the
            # 'is the cls head dead?' diagnostic).
            self._accum_lane_scores(final_preds)
            # Stash per-lane predictions so update_metrics can compute the
            # CLRKDNet-style curve F1 against the per-lane GT lane_targets
            # (the merged pixel-IoU saturates ~0.09 and hides real progress).
            self._last_final_preds = final_preds.detach()

            mask = batch_lanes_to_mask(
                final_preds, img_h=self.args.imgsz, img_w=self.args.imgsz,
                num_points=72, line_width=8,
                # conf_threshold=None -> NO absolute threshold; pure top-N
                # by score ranking, so IoU tracks the cls ranking instead
                # of freezing behind a cutoff every prior clears at ~0.5.
                conf_threshold=None,
                max_lanes=8,
            )
            # Diagnostic: fraction of the predicted mask that is lane. If
            # this is ~0 the rasterizer drew nothing (the length-scaling
            # bug); after the fix it should be non-zero AND change across
            # epochs as the geometry learns. Surfaced in _finalize.
            try:
                acc = getattr(self, '_lane_score_acc', None)
                if acc is not None:
                    acc['mask_px'] = acc.get('mask_px', 0.0) + float(mask.float().mean())
                    acc['mask_n'] = acc.get('mask_n', 0) + 1
            except Exception:  # noqa: BLE001
                pass
        else:
            mask_thr_tensor = self.mask_thr.to(device=seg_part.device)
            mask = (torch.sigmoid(seg_part) > mask_thr_tensor).float()

        return outputs, mask

    def _prepare_batch(self, si, batch, detection_indices):
        """Prepares a batch for training or inference by applying transformations."""
        idx_detection = detection_indices[batch["batch_idx"][detection_indices] == si]
        cls_detection = batch["cls"][idx_detection].squeeze(-1)
        bbox_detection = batch["bboxes"][idx_detection]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls_detection):
            bbox_detection = ops.xywh2xyxy(bbox_detection)  # target boxes
            bbox_detection[..., [0, 2]] *= ori_shape[1]  # native-space pred
            bbox_detection[..., [1, 3]] *= ori_shape[0]  # native-space pred
        return {"cls": cls_detection, "bbox": bbox_detection, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch with transformed bounding boxes and class labels."""
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        return predn.float()

    def update_metrics(self, preds_list, batch):
        """Metrics."""
        preds = preds_list[0]
        merge_mask = preds_list[1]
        type_task = batch['type_task'][0]
        detection_classes = torch.tensor(type_task['detection'])
        detection_indices = torch.where(torch.isin(batch["cls"], detection_classes.to(device=batch["cls"].device)))[0]
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )

            pbatch_detection= self._prepare_batch(si, batch, detection_indices)
            cls, bbox = pbatch_detection.pop("cls"), pbatch_detection.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch_detection)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            ### JW segmentataion evaluate
            _, nc, _, _ = merge_mask.shape
            for seg_nc in range(nc):
                task_name = self.data['names'][self.data['type_task']['segmentation'][seg_nc]]
                pred_mask = merge_mask[si][seg_nc].squeeze()
                gt_mask = batch['merge_mask'][si][seg_nc].to(device=pred_mask.device)
                self.seg_metrics[task_name].reset()

                if gt_mask.shape != pred_mask.shape:
                    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
                    gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 320, 320)
                    pred_mask = F.interpolate(
                        pred_mask,
                        size=gt_mask.shape[-2:],
                        mode='bilinear',
                    )
                    pred_mask = pred_mask.squeeze(0).squeeze(0)
                    gt_mask = gt_mask.squeeze(0).squeeze(0)

                self.seg_metrics[task_name].addBatch(pred_mask.cpu(), gt_mask.cpu())
                self.seg_result[task_name]['pixacc'].update(self.seg_metrics[task_name].pixelAccuracy())
                self.seg_result[task_name]['subacc'].update(self.seg_metrics[task_name].lineAccuracy())
                self.seg_result[task_name]['IoU'].update(self.seg_metrics[task_name].IntersectionOverUnion())
                self.seg_result[task_name]['mIoU'].update(self.seg_metrics[task_name].meanIntersectionOverUnion())
                if self.args.plots:
                    self.plot_masks[task_name].append(pred_mask.cpu())  ### JW TODO Need to adapt to the segmentation task plot

            # CLRKDNet-style per-lane curve F1 (the un-saturated yardstick).
            # Needs per-lane pred (stashed in postprocess) + per-lane GT
            # (batch['lane_targets']). Handles BOTH the 78-D polyline head
            # and the 16-D bezier head (the decode dispatches on width), so
            # the two representations can be ranked on the SAME metric
            # (pixel-IoU saturates ~0.09 and can't tell them apart).
            lt = batch.get('lane_targets', None) if isinstance(batch, dict) else None
            fp_preds = getattr(self, '_last_final_preds', None)
            if lt is not None and fp_preds is not None and si < len(fp_preds) and si < len(lt):
                try:
                    pr = fp_preds[si].cpu().numpy()
                    gt = lt[si].cpu().numpy() if hasattr(lt[si], 'cpu') else lt[si]
                    if pr.shape[-1] in (78, 16):  # polyline or bezier
                        from ultralytics.models.utils.lane_curve_f1 import lane_curve_tp_fp_fn
                        acc = getattr(self, '_curve_f1_acc', None)
                        if acc is None:
                            acc = {'tp': 0, 'fp': 0, 'fn': 0,
                                   'iou_sum': 0.0, 'n_gt': 0}
                            self._curve_f1_acc = acc
                        # S1.1: honor a decode confidence threshold so the F1
                        # the validator reports matches a thresholded decode
                        # (env LANE_EVAL_TAU, default 0 = old top-8 behavior).
                        import os as _os
                        _tau = float(_os.environ.get('LANE_EVAL_TAU', '0') or 0)
                        tp, fpp, fn, iou_sum, ngt = lane_curve_tp_fp_fn(
                            pr, gt, conf_tau=_tau)
                        acc['tp'] += tp; acc['fp'] += fpp; acc['fn'] += fn
                        acc['iou_sum'] += iou_sum; acc['n_gt'] += ngt
                except Exception:  # noqa: BLE001 - never let a diagnostic crash val
                    pass

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix_detection.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch_detection["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def get_stats(self):
        """Returns metrics statistics and results dictionary.

        Surfaces BOTH detection AND segmentation (lane) metrics so the
        per-epoch trainer log + results.csv + our per-epoch metrics
        table callback can all read them from a single dict.

        Lane metrics use the `metrics/<name>(lane)` key convention so
        downstream callbacks can pick them out alongside the detection
        `metrics/<name>(B)` keys.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc['detection'])
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc['detection'])
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics['detection'].process(**stats)
        out = dict(self.metrics['detection'].results_dict)

        # Merge segmentation (lane) metrics. self.seg_result is populated
        # per-batch in update_metrics; AverageMeter.avg returns 0 if no
        # batches ran (defensive default for early-startup get_stats).
        seg_index = self.data.get('type_task', {}).get('segmentation', [])
        names = self.data.get('names', {})
        for i in seg_index:
            task_name = names.get(i, str(i))
            results = self.seg_result.get(task_name)
            if results is None:
                continue
            for stat_key, meter in results.items():
                # e.g. 'metrics/IoU(lane)', 'metrics/pixacc(lane)'
                out[f'metrics/{stat_key}({task_name})'] = float(meter.avg)

        # Curve F1 - the un-saturated lane yardstick. Emit + reset the
        # per-pass accumulator. Lets us see whether lane quality keeps
        # improving even when the merged pixel-IoU plateaus (~0.09).
        cacc = getattr(self, '_curve_f1_acc', None)
        if cacc is not None and (cacc['tp'] + cacc['fp'] + cacc['fn']) > 0:
            from ultralytics.models.utils.lane_curve_f1 import f1_from_counts
            r = f1_from_counts(cacc['tp'], cacc['fp'], cacc['fn'])
            # Un-saturated companion: mean best-match IoU over all GT lanes.
            # Climbs smoothly 0 -> 0.5 -> ... so it shows geometry sharpening
            # EVEN WHILE F1@0.5 is still pinned at 0 (a weak model whose
            # lanes are close-ish but not yet within the 0.5 bar). This is
            # the number to watch for "is the model still learning lanes?".
            cur_iou = (cacc['iou_sum'] / cacc['n_gt']) if cacc['n_gt'] > 0 else 0.0
            LOGGER.info(
                f"[lane-f1] curve F1={r['f1']:.4f} P={r['precision']:.4f} "
                f"R={r['recall']:.4f} (tp={r['tp']} fp={r['fp']} fn={r['fn']}) "
                f"@IoU>=0.5 | mean-best-IoU(curveIoU)={cur_iou:.4f}  "
                f"(F1@0.5 = strict top-8 detection metric; curveIoU = "
                f"un-saturated geometry over the broad prior pool)"
            )
            out['metrics/lane_f1(lane)'] = float(r['f1'])
            out['metrics/lane_precision(lane)'] = float(r['precision'])
            out['metrics/lane_recall(lane)'] = float(r['recall'])
            out['metrics/lane_curveIoU(lane)'] = float(cur_iou)
            self._curve_f1_acc = None  # reset for the next pass

        # Phase 1: finalize + print the lane pos-score histogram and add a
        # score-spread metric. This is the visual 'are the 192 priors still
        # clustered at 0.5 and not moving?' check plus the machine-readable
        # 'is the lane head dead?' signal (lane_score_std).
        out.update(self._finalize_lane_scores())

        # COMBINED multi-task fitness. The default 'fitness' from
        # DetMetrics is detection-only (0.1*mAP50 + 0.9*mAP50-95), so
        # best.pt and EarlyStopping lock onto the detection peak (~epoch
        # 20 in NB88/89) and ignore the slower lane branch entirely - the
        # exact "best at epoch 20, then 200 wasted epochs" symptom. Add a
        # lane-IoU bonus on top of the detection base so:
        #   - best.pt updates whenever EITHER det OR lane improves
        #   - EarlyStopping waits until BOTH plateau
        #   - detection is never traded away (it's the base term)
        # Weight 0.5 keeps detection dominant while letting the lane signal
        # keep fitness climbing past the detection plateau. Env-tunable.
        #
        # LANE_FITNESS_METRIC selects WHICH lane signal drives best.pt /
        # EarlyStopping:
        #   'curvef1' (default) = 0.5*curveIoU + 0.5*lane_f1 -- the metrics that
        #       ACTUALLY MOVE on this head (NB106: curveIoU 0.52->0.61, F1
        #       0.15->0.26). pixel-IoU saturates ~0.09 and is a dead signal, so
        #       tracking it made best.pt lock onto a stale checkpoint while the
        #       geometry kept improving. This keeps the geometrically-best ckpt.
        #   'pixiou' = the old metrics/IoU(lane) behavior (back-compat).
        try:
            import os as _os
            w_lane = float(_os.environ.get('LANE_FITNESS_WEIGHT', '0.5'))
            det_fit = float(out.get('fitness', 0.0) or 0.0)
            metric = _os.environ.get('LANE_FITNESS_METRIC', 'curvef1').strip().lower()
            if metric == 'pixiou':
                lane_term = float(out.get('metrics/IoU(lane)', 0.0) or 0.0)
            else:  # 'curvef1': blend the two un-saturated, moving signals
                _ci = float(out.get('metrics/lane_curveIoU(lane)', 0.0) or 0.0)
                _f1 = float(out.get('metrics/lane_f1(lane)', 0.0) or 0.0)
                lane_term = 0.5 * _ci + 0.5 * _f1
            out['fitness'] = det_fit + w_lane * lane_term
        except Exception:  # noqa: BLE001
            pass
        return out

    # ------------------------------------------------------------------
    # Phase 1: lane pos-score diagnostics
    # ------------------------------------------------------------------
    def _accum_lane_scores(self, final_preds):
        """Accumulate per-prior pos-scores across the val pass (called per
        batch in postprocess). Lazily (re)initializes the accumulator."""
        import numpy as _np
        acc = getattr(self, '_lane_score_acc', None)
        if acc is None or acc.get('done'):
            acc = {'hist': _np.zeros(20, dtype=_np.int64),
                   'min': 1.0, 'max': 0.0, 'sum': 0.0, 'sqsum': 0.0,
                   'n': 0, 'done': False,
                   # GEOMETRY accumulators. The frozen-IoU root-cause
                   # hypothesis is "cls learns but geometry (reg_layers)
                   # does NOT, so the rasterized eval mask is the fixed
                   # prior-anchor fan." These per-epoch means/stds of the
                   # geometry fields make that testable: if they are
                   # byte-identical across epochs, the geometry is frozen
                   # at the anchor init and that IS why IoU never moves.
                   'g_sum': _np.zeros(3, dtype=_np.float64),     # start_x, theta, length
                   'g_sqsum': _np.zeros(3, dtype=_np.float64),
                   'g_n': 0,
                   # BEZIER-only: mean t_start / t_end of the predicted
                   # curves (16-D slots [10]/[11]). The bezier head used to
                   # init t_start=0.5, t_end=0.55 -> each curve rendered only
                   # the middle 5% of its span -> near-zero mask overlap ->
                   # stuck IoU. After the bias-init fix these should read
                   # ~0.02 / ~0.98 (full span) from epoch 1. span = t_end -
                   # t_start is the headline number: if it stays small the
                   # curves are stubs and IoU can't climb.
                   'tr_sum': _np.zeros(2, dtype=_np.float64),     # t_start, t_end
                   'tr_n': 0}
            self._lane_score_acc = acc
        with torch.no_grad():
            fp = final_preds.detach().float()
            logits = fp[..., :2].reshape(-1, 2)
            s = torch.softmax(logits, dim=1)[:, 1]
            h = torch.histc(s, bins=20, min=0.0, max=1.0).cpu().numpy().astype('int64')
            acc['hist'] += h
            acc['min'] = min(acc['min'], float(s.min()))
            acc['max'] = max(acc['max'], float(s.max()))
            acc['sum'] += float(s.sum())
            acc['sqsum'] += float((s * s).sum())
            acc['n'] += int(s.numel())
            # Geometry fields. 78-D polyline: [3]=start_x [4]=theta [5]=length.
            # 16-D bezier: use [3]=P1.x [4]=P2.x [5]=P3.x as a geometry proxy.
            D = fp.shape[-1]
            if D >= 6:
                g = fp[..., 3:6].reshape(-1, 3).cpu().numpy().astype(_np.float64)
                acc['g_sum'] += g.sum(axis=0)
                acc['g_sqsum'] += (g * g).sum(axis=0)
                acc['g_n'] += g.shape[0]
            # Bezier t-range (16-D only): slots [10]=t_start, [11]=t_end.
            if D == 16:
                tr = fp[..., 10:12].reshape(-1, 2).cpu().numpy().astype(_np.float64)
                acc['tr_sum'] += tr.sum(axis=0)
                acc['tr_n'] += tr.shape[0]

    def _finalize_lane_scores(self) -> dict:
        """Compute spread stats, print a histogram, mark the pass done so
        the next val pass starts fresh. Returns metric keys to merge."""
        acc = getattr(self, '_lane_score_acc', None)
        if not acc or acc['n'] == 0:
            return {}
        n = acc['n']
        mean = acc['sum'] / n
        var = max(0.0, acc['sqsum'] / n - mean * mean)
        std = var ** 0.5
        spread = acc['max'] - acc['min']
        # Coarse 20-bin histogram, rendered as a compact bar line so it's
        # eyeball-able in the training log.
        hist = acc['hist']
        smin = acc['min']
        smax = acc['max']
        peak = max(1, int(hist.max()))
        bars = ''.join(
            ' .:-=+*#%@'[min(9, int(9 * c / peak))] for c in hist
        )
        LOGGER.info(
            f'[lane-score] n={n} min={smin:.4f} max={smax:.4f} '
            f'mean={mean:.4f} std={std:.5f} spread={spread:.5f}'
        )
        LOGGER.info(f'[lane-score] hist[0..1] |{bars}|  (peak={peak})')
        if std < 1e-3:
            LOGGER.info(
                '[lane-score] WARNING: pos-score std < 1e-3 -> cls head looks '
                'DEAD (all priors clustered; decoded IoU will not move).'
            )
        # GEOMETRY stats: per-epoch mean of (start_x, theta, length) over
        # all priors. If these are byte-identical epoch-to-epoch, the
        # regression branch is frozen at the anchor init -> THAT is why
        # decoded IoU never moves (cls can spread all it wants, the mask
        # geometry never changes).
        out_geom = {}
        if acc.get('g_n', 0) > 0:
            gmean = acc['g_sum'] / acc['g_n']
            gvar = (acc['g_sqsum'] / acc['g_n']) - gmean * gmean
            gstd = (gvar.clip(min=0.0)) ** 0.5
            LOGGER.info(
                f'[lane-geom] start_x mean={gmean[0]:.5f} std={gstd[0]:.5f} | '
                f'theta mean={gmean[1]:.5f} std={gstd[1]:.5f} | '
                f'length mean={gmean[2]:.5f} std={gstd[2]:.5f}  '
                f'(if these are constant across epochs -> geometry FROZEN)'
            )
            out_geom = {
                'metrics/lane_startx_mean(lane)': float(gmean[0]),
                'metrics/lane_theta_mean(lane)': float(gmean[1]),
                'metrics/lane_length_mean(lane)': float(gmean[2]),
                'metrics/lane_startx_std(lane)': float(gstd[0]),
            }
        # BEZIER t-range (16-D heads only). span = t_end - t_start is the
        # decisive number for the bezier-stub bug: it must be ~0.96 (full
        # curve) not ~0.05 (middle-5% stub). If span is small the bias-init
        # fix did not take / the head collapsed the validity range again.
        if acc.get('tr_n', 0) > 0:
            tr_mean = acc['tr_sum'] / acc['tr_n']
            span = float(tr_mean[1] - tr_mean[0])
            LOGGER.info(
                f'[bezier-trange] t_start mean={tr_mean[0]:.4f} '
                f't_end mean={tr_mean[1]:.4f} span={span:.4f}  '
                f'(span ~0.96 => full curve drawn; ~0.05 => 5% stub bug)'
            )
            out_geom['metrics/lane_bezier_span(lane)'] = span
        # Predicted-mask lane-pixel fraction (decisive: did the rasterizer
        # actually draw lanes this epoch, and does it change?).
        if acc.get('mask_n', 0) > 0:
            mask_frac = acc['mask_px'] / acc['mask_n']
            LOGGER.info(
                f'[lane-mask] predicted lane-pixel fraction = {mask_frac:.5f}  '
                f'(0 => rasterizer drew nothing; should grow as geometry learns)'
            )
            out_geom['metrics/lane_mask_frac(lane)'] = float(mask_frac)
        acc['done'] = True
        return {
            'metrics/lane_score_std(lane)': float(std),
            'metrics/lane_score_spread(lane)': float(spread),
            'metrics/lane_score_mean(lane)': float(mean),
            **out_geom,
        }

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics['detection'].confusion_matrix = self.confusion_matrix

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics['detection'].keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics['detection'].mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc['detection'] > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics['detection'].class_result(i))
                )

        pf = '%22s' + ('%11s' + '%11.3g') * 4
        seg_index = self.data['type_task']['segmentation']
        for i in seg_index:
            key_values = [(key, value.avg) for key, value in self.seg_result[self.data['names'][i]].items()]
            LOGGER.info(pf % (self.data['names'][i], *sum(key_values, ())))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix_detection.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result.

        Multi-task fix: this validator's `postprocess` returns a 2-tuple
        `(detection_outputs_list, lane_mask)`. `output_to_target` expects
        ONLY the per-image detection-tensor list, so passing the whole
        tuple made it iterate `(outputs, mask)` and index the `outputs`
        LIST with a tuple -> `TypeError: list indices must be integers
        or slices, not tuple`. That crashed final_eval AFTER a full
        30-epoch run completed. Extract the detection list, and wrap the
        whole thing so a pure-visualization error can never discard a
        finished training run.
        """
        if isinstance(preds, tuple) and len(preds) == 2:
            det = preds[0]  # (outputs_list, lane_mask) -> detection list
        else:
            det = preds
        try:
            plot_images(
                batch["img"],
                *output_to_target(det, max_det=self.args.max_det),
                paths=batch["im_file"],
                fname=self.save_dir / f"val_batch{ni}_pred.jpg",
                names=self.names,
                on_plot=self.on_plot,
            )  # pred
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                f"plot_predictions skipped (non-fatal): {type(e).__name__}: {e}"
            )