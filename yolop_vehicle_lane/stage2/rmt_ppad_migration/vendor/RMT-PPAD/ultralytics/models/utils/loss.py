# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss, DiceLoss, FocalLossV1, tversky, DiceLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


# ---------------------------------------------------------------------------
# Sprint-1 lane-head lever helpers (NEXT_STEPS sec 5). Module-level so they are
# importable/testable in isolation. All are no-ops unless their env flag is set.
# ---------------------------------------------------------------------------

def _lane_match_line_iou(reg_pred, reg_targets, img_w):
    """Per-match line-IoU (aligned) for the IoU-aware cls target (S1.3).

    reg_pred/reg_targets: (M, 72) x-grids in PIXELS. Returns (M,) IoU in [0,1],
    reusing the same line_iou the regression loss uses so the cls target and
    the geometry loss agree.
    """
    from ultralytics.models.utils.lane_losses import line_iou
    return line_iou(reg_pred, reg_targets, img_w, length=15, aligned=True)


def _qfl_loss(cls_pred, soft_target, gamma=2.0, eps=1e-6):
    """Quality Focal Loss (Generalized Focal Loss, NeurIPS'20) for S1.3.

    cls_pred: (N, 2) logits [neg, pos]. soft_target: (N,) in [0,1] = the line-
    IoU of the matched prior (0 for negatives). The positive-channel sigmoid is
    regressed toward the soft target with a |target - p|^gamma modulation, so
    the score learns to encode geometric quality (spreading score_std).
    """
    p = torch.sigmoid(cls_pred[:, 1]).clamp(eps, 1 - eps)
    scale = (soft_target - p).abs() ** gamma
    bce = -(soft_target * torch.log(p) + (1 - soft_target) * torch.log(1 - p))
    return scale * bce


def _y_reweighted_liou(reg_pred, reg_targets, img_w, liou_fn, mode='near'):
    """S1.4: y-reweighted line-IoU loss.

    x-grid index 0 = img_h (BOTTOM = near camera); index increases UPWARD to far
    (verified vs lane_mask_from_target.offsets_ys = arange(img_h, -1, -step)).
    So reg[:, :half] is the NEAR half and reg[:, half:] is the FAR half. 'near'
    upweights the near rows (where length shrinks / the model gives up); 'angle'
    weights near a touch more but milder. liou_fn returns (1 - IoU); we blend the
    near-half and far-half liou. Cheap and monotone.
    """
    n = reg_pred.shape[1]
    half = n // 2
    near = liou_fn(reg_pred[:, :half], reg_targets[:, :half], img_w, length=15)
    far = liou_fn(reg_pred[:, half:], reg_targets[:, half:], img_w, length=15)
    if mode == 'near':
        return 0.35 * far + 0.65 * near       # 1.85x weight on near rows
    # 'angle': weight near a bit more but milder than 'near'
    return 0.45 * far + 0.55 * near


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0
    ):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with
        preset cost gains. Supports auxiliary losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict): Coefficients for different loss components.
            aux_loss (bool): Use auxiliary losses from each decoder layer.
            use_fl (bool): Use FocalLoss.
            use_vfl (bool): Use VarifocalLoss.
            use_uni_match (bool): Use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc['detection'] + 1), dtype=torch.int64, device=targets.device) ### JW use detection nc
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """Computes bounding box and GIoU losses for predicted and ground truth bounding boxes."""
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """Get auxiliary losses."""
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """Get losses."""
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc['detection'], device=pred_scores.device, dtype=gt_cls.dtype) ### JW use detection nc
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        # if masks is not None and gt_mask is not None:
        #     loss.update(self._get_loss_mask(masks, gt_mask, match_indices, postfix))
        return loss

    def forward(self, pred_bboxes, pred_scores, batch, postfix="", **kwargs):
        """
        Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape [l, b, query, 4].
            pred_scores (torch.Tensor): Predicted class scores, shape [l, b, query, num_classes].
            batch (dict): Batch information containing:
                cls (torch.Tensor): Ground truth classes, shape [num_gts].
                bboxes (torch.Tensor): Ground truth bounding boxes, shape [num_gts, 4].
                gt_groups (List[int]): Number of ground truths for each image in the batch.
            postfix (str): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict): Computed losses, including main and auxiliary (if enabled).

        Note:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices



class MTDETRDLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """
    def __init__(
        self, nc=80, use_vfl=False, ema_decay=0.99, eps=1e-6,
        lane_only_mode=False, bezier_mode=False, with_lcm=False,
    ):
        super().__init__(nc=nc, use_vfl=use_vfl)
        self.FocalLoss = FocalLossV1()
        self.TL = tversky()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dl = DiceLoss()
        # P5: lane-only mode swaps the drivable+lane mask losses for the
        # 4-term CLRKDNet lane loss (cls focal + xytl smooth-L1 + line-IoU
        # + binary aux seg). seg_mask in this mode is a dict from
        # LaneSegHead, not the tuple from TransformerSegmentationDecoder.
        # B5: bezier_mode is a sub-mode of lane-only that further swaps
        # the polyline 4-term loss for the Bezier 4-term loss
        # (cls focal + bezier_geom smooth-L1 + bezier_line_iou + aux seg).
        self.lane_only_mode = lane_only_mode
        self.bezier_mode = bezier_mode
        self.with_lcm = with_lcm
        # Phase 1: lane prior<->GT matching scheme. 'hungarian' (1-to-1,
        # deterministic) fixes the cls collapse that 'dynamic_k' causes;
        # see lane_losses.hungarian_assign. Read from env so the deep
        # Ultralytics call stack doesn't need a new plumbed kwarg; the
        # train scripts set LANE_MATCH before importing ultralytics.
        import os as _os
        self.lane_match = _os.environ.get('LANE_MATCH', 'hungarian').strip().lower()
        if self.lane_match not in ('hungarian', 'dynamic_k'):
            self.lane_match = 'hungarian'
        print(f'[MTDETRDLoss] lane matching scheme = {self.lane_match}', flush=True)
        # Diagnostic knob: xytl smooth-l1 diff clamp (px). 'none' disables.
        _clamp_env = _os.environ.get('LANE_DIFF_CLAMP', '100').strip().lower()
        self.xytl_diff_clamp = None if _clamp_env in ('none', '0', 'off') else float(_clamp_env)
        print(f'[MTDETRDLoss] xytl diff clamp = {self.xytl_diff_clamp}', flush=True)
        # ---- Sprint-1 lane-head levers (NEXT_STEPS sec 5), env-driven like the
        # knobs above so the NB105 ablation grid can flip them per-row without
        # plumbing kwargs through the Ultralytics call stack. All OFF by default
        # => byte-identical to the NB101 recipe when unset.
        #   S1.3 LANE_IOU_CLS: soft cls target = line-IoU(matched prior, GT) in
        #        [0,1] (QFL-style) instead of a hard 1. Spreads score_std (0.09)
        #        so the S0.3 threshold becomes stable. Value = QFL gamma (2.0).
        #   S1.2 LANE_SMOOTH_W: weight on a 2nd-difference (curvature) penalty on
        #        each matched lane's x-offsets. OPTIONAL row (S0.4 showed near is
        #        already smoother than far, so this is a re-check, not a given).
        #   S1.4 LANE_Y_REWEIGHT: 'near' weights the xytl diff heavier on
        #        near-camera rows (bottom); 'angle' normalizes by local dx/dy;
        #        'none' = uniform (current). Targets the length-shrink 0.38->0.28.
        self.lane_iou_cls = _os.environ.get('LANE_IOU_CLS', 'off').strip().lower()
        self.lane_iou_cls_on = self.lane_iou_cls not in ('off', '0', 'none', '')
        self.lane_iou_cls_gamma = (float(self.lane_iou_cls) if self.lane_iou_cls
                                   not in ('on', 'off', '0', 'none', '') else 2.0)
        self.lane_smooth_w = float(_os.environ.get('LANE_SMOOTH_W', '0') or 0)
        self.lane_y_reweight = _os.environ.get('LANE_Y_REWEIGHT', 'none').strip().lower()
        #   S1.4b LANE_LEN_HINGE_W: weight on an ASYMMETRIC "too-short" hinge on
        #        the predicted length field (col 5). The xytl smooth-L1 below
        #        supervises length SYMMETRICALLY and as 1/4 of a mixed-scale
        #        mean, so the model freely SHORTENS lanes (length_mean shrank
        #        0.155->0.094 in NB107) to dodge x-error on hard rows. This term
        #        penalizes ONLY (gt_len - pred_len)_+ (normalized to [0,1] by
        #        n_strips), directly fighting the shrink. 0 = off (default).
        self.lane_len_hinge_w = float(_os.environ.get('LANE_LEN_HINGE_W', '0') or 0)
        print(f'[MTDETRDLoss] S1 levers: iou_cls={self.lane_iou_cls_on}'
              f'(g={self.lane_iou_cls_gamma}) smooth_w={self.lane_smooth_w} '
              f'y_reweight={self.lane_y_reweight} len_hinge_w={self.lane_len_hinge_w}',
              flush=True)
        # AUX dense seg (training-only): drivable + lane dense supervision
        # re-attached to fight detection gradient domination. On/off + weights
        # via env (set by the train script before ultralytics import, like
        # LANE_MATCH). Ignored automatically at val (the decoder stashes a
        # None aux output in eval mode).
        self.use_aux_seg = _os.environ.get('USE_AUX_SEG', '') == '1'
        self.aux_drivable_weight = float(_os.environ.get('AUX_DRIVABLE_WEIGHT', '0.5'))
        self.aux_lane_weight = float(_os.environ.get('AUX_LANE_WEIGHT', '0.5'))
        if self.use_aux_seg:
            print(f'[MTDETRDLoss] AUX dense seg ON: drivable_w={self.aux_drivable_weight} '
                  f'lane_w={self.aux_lane_weight} (training-only)', flush=True)
        if lane_only_mode and not bezier_mode:
            from ultralytics.models.utils.lane_losses import (
                FocalLossForLane, assign, liou_loss, lane_iou_loss,
            )
            self.lane_cls_loss_fn = FocalLossForLane(alpha=0.25, gamma=2.0)
            self.lane_assign = assign
            # S2.A: angle-aware LaneIoU (CLRerNet) vs the fixed-band line-IoU.
            # LANE_IOU_TYPE=laneiou widens the IoU band where the lane is steep
            # (near-field) so it penalizes shape error proportional to local
            # angle -- the published fix for our flat-curveIoU + near-field
            # jaggedness. Default 'line' = unchanged.
            self.lane_iou_type = _os.environ.get('LANE_IOU_TYPE', 'line').strip().lower()
            self.lane_iou_loss_fn = (lane_iou_loss if self.lane_iou_type == 'laneiou'
                                     else liou_loss)
            print(f'[MTDETRDLoss] lane IoU type = {self.lane_iou_type}', flush=True)
            # Weights from CLRKDNet's configs/DLA_CULane.py, scaled DOWN
            # by 5x for the joint detection+lane setup. CLRKDNet's
            # original 2.0/0.5/2.0/1.0 were tuned for lane-only training
            # on a ~5M-param model; in our 35M-param RT-DETR + lane
            # joint setup they made the lane gradient dominate the
            # shared backbone at init (NB88 row 1 saw ll_seg EMA
            # 16 -> 240 over 500 warmup batches). The 5x reduction
            # preserves CLRKDNet's relative balance (cls:xytl:iou:seg
            # = 4:1:4:2) while letting detection share the backbone
            # updates fairly. P8 ablation can revisit if lane IoU is
            # too low at convergence.
            # Diagnostic knobs (env-driven, like LANE_MATCH) so NB95/NB96
            # can ablate the geometry-supervision strength WITHOUT editing
            # code. LANE_WEIGHT_PRESET: 'cut5x' (default, the joint-stable
            # 5x reduction) or 'clrkd' (CLRKDNet's full cls=2 xytl=0.5
            # iou=2 seg=1 - stronger geometry gradient, the prime suspect
            # for why reg_layers wasn't learning).
            preset = _os.environ.get('LANE_WEIGHT_PRESET', 'cut5x').strip().lower()
            scale = 1.0 if preset == 'clrkd' else 0.2
            self.lane_cls_weight = 2.0 * scale
            self.lane_xytl_weight = 0.5 * scale
            self.lane_iou_weight = 2.0 * scale
            self.lane_seg_weight = 1.0 * scale
            print(f'[MTDETRDLoss] lane weight preset={preset} '
                  f'(cls={self.lane_cls_weight} xytl={self.lane_xytl_weight} '
                  f'iou={self.lane_iou_weight} seg={self.lane_seg_weight})', flush=True)
        elif lane_only_mode and bezier_mode:
            # B5: import bezier loss helpers. Same 5x weight haircut as
            # the polyline path so the joint-training balance carries
            # over. The geom L1 part is now in NORMALIZED coordinate
            # space (control points in [0, 1]) so the diff_clamp is
            # tighter than the polyline +/-100 px clamp.
            from ultralytics.models.utils.bezier_losses import (
                FocalLossForLane, assign_bezier,
                lane_bezier_geom_loss, bezier_liou_loss,
                complexity_penalty_loss,
            )
            self.lane_cls_loss_fn = FocalLossForLane(alpha=0.25, gamma=2.0)
            self.bezier_assign = assign_bezier
            self.bezier_geom_loss_fn = lane_bezier_geom_loss
            self.bezier_iou_loss_fn = bezier_liou_loss
            self.complexity_loss_fn = complexity_penalty_loss
            self.lane_cls_weight = 2.0 * 0.2       # 0.4
            self.bezier_geom_weight = 1.0 * 0.2    # 0.2 (was 0.1 for xytl; geom touches
                                                   #       all 4 cps + validity so kept higher)
            self.lane_iou_weight = 2.0 * 0.2       # 0.4
            self.lane_seg_weight = 1.0 * 0.2       # 0.2
            self.complexity_weight = 0.01          # appendix sec 5.3

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None, seg_mask=None, seg_batch=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.
            seg_mask (list) storage predicted segmentation mask and aux_list

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})


        ### JW segmentation loss.
        if self.lane_only_mode:
            # P5 / B5: lane-only path. seg_mask is [head_dict, None] (the
            # task loss() wrapper still passes [seg_masks, aux_list]). We
            # only use the dict.
            lane_seg_output = seg_mask[0]
            if self.bezier_mode:
                total_loss = self._compute_bezier_only_loss(
                    lane_seg_output, seg_batch, total_loss,
                )
            else:
                total_loss = self._compute_lane_only_loss(
                    lane_seg_output, seg_batch, total_loss,
                )
        else:
            gt_seg = seg_batch['merge_mask'].float()
            loss_drivable_fl = self.FocalLoss(seg_mask[0][:, 0, :, :],gt_seg[:, 0, :, :]) * self.fl_weight
            loss_drivable_tv = self.bce_loss(seg_mask[0][:, 0, :, :],gt_seg[:, 0, :, :]) * self.tversky_weight
            loss_lane_fl = self.FocalLoss(seg_mask[0][:, 1, :, :], gt_seg[:, 1, :, :]) * self.fl_weight
            loss_lane_tv = self.TL(seg_mask[0][:, 1, :, :], gt_seg[:, 1, :, :]) * self.tversky_weight
            total_loss.update({"da_fl_loss": loss_drivable_fl, "da_tversky_loss":loss_drivable_tv, "ll_fl_loss": loss_lane_fl, "ll_tversky_loss": loss_lane_tv})

        # AUX dense seg (training-only). seg_mask[2] is the decoder's stashed
        # aux logits (None at eval or when off). Adds drivable+lane dense
        # terms keyed so tasks.py routes drivable -> da_seg, lane -> ll_seg.
        if self.use_aux_seg and len(seg_mask) > 2 and seg_mask[2] is not None:
            total_loss = self._compute_aux_seg_loss(seg_mask[2], seg_batch, total_loss)

        return total_loss

    def _compute_aux_seg_loss(self, aux_logits, seg_batch, total_loss):
        """Step 4: dense drivable + lane aux loss (Focal + Tversky).

        aux_logits: (B, C, H, W). Channel 0 = drivable, 1 = lane (matches the
        legacy da/ll convention so GT masks line up). GT comes from seg_batch:
        'lane_seg_mask' (always present in lane-only mode) and
        'drivable_seg_mask' (only after the subset is rebuilt with drivable
        masks). Each term is gated on its GT being present, so this degrades
        gracefully to lane-only aux until drivable GT lands.
        """
        import torch.nn.functional as _F

        def _hw(gt):
            g = gt.float()
            return g[:, 0] if g.dim() == 4 else g  # (B, H, W) in {0,1}

        def _match(pred_hw, gt_hw):
            # upsample pred logits to the GT mask resolution if needed
            if pred_hw.shape[-2:] != gt_hw.shape[-2:]:
                pred_hw = _F.interpolate(
                    pred_hw.unsqueeze(1), size=gt_hw.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).squeeze(1)
            return pred_hw, gt_hw

        C = aux_logits.shape[1]
        lane_gt = seg_batch.get('lane_seg_mask', None) if isinstance(seg_batch, dict) else None
        driv_gt = seg_batch.get('drivable_seg_mask', None) if isinstance(seg_batch, dict) else None

        if lane_gt is not None and C >= 2:
            p, g = _match(aux_logits[:, 1], _hw(lane_gt))
            total_loss['aux_ll_loss'] = (self.FocalLoss(p, g) + self.TL(p, g)) * self.aux_lane_weight
        if driv_gt is not None and C >= 1:
            p, g = _match(aux_logits[:, 0], _hw(driv_gt))
            total_loss['aux_da_loss'] = (self.FocalLoss(p, g) + self.TL(p, g)) * self.aux_drivable_weight
        return total_loss

    def _compute_lane_only_loss(self, lane_seg_output, seg_batch, total_loss):
        """P5: compute the 4-term CLRKDNet lane loss given LaneSegHead's
        forward output dict and the dataloader-provided lane targets.

        lane_seg_output: dict {'lane_output': X}
            X is itself a dict {'predictions_lists': [tensor(B, P, 78), ...],
                                'seg': tensor(B, 2, H_s, W_s)} (training) OR
            a single tensor (B, P, 78) (eval).
        seg_batch: dict from task.loss() carrying
            'lane_targets' (B, max_lanes, 78) and
            'lane_seg_mask' (B, 1, H, W) for aux supervision.
        """
        import torch.nn.functional as F
        lane_output = lane_seg_output['lane_output']
        if not isinstance(lane_output, dict):
            # Eval mode - skip loss computation (validator handles it).
            total_loss.update({
                'lane_cls_loss': torch.tensor(0.0, device=self.device),
                'lane_xytl_loss': torch.tensor(0.0, device=self.device),
                'lane_iou_loss': torch.tensor(0.0, device=self.device),
                'lane_seg_aux_loss': torch.tensor(0.0, device=self.device),
            })
            return total_loss

        predictions_lists = lane_output['predictions_lists']
        lane_seg = lane_output['seg']
        lane_targets = seg_batch['lane_targets']  # (B, max_lanes, 78)
        img_w = img_h = 640
        # n_strips = num_points - 1 = 71 for num_points=72
        n_strips = predictions_lists[0].shape[-1] - 6 - 1

        device = predictions_lists[0].device
        batch_size = lane_targets.shape[0]
        refine_layers = len(predictions_lists)

        cls_loss_sum = torch.tensor(0.0, device=device)
        xytl_loss_sum = torch.tensor(0.0, device=device)
        iou_loss_sum = torch.tensor(0.0, device=device)

        for stage in range(refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, lane_targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss_sum = cls_loss_sum + self.lane_cls_loss_fn(cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = self.lane_assign(
                        predictions, target, img_w, img_h,
                        match=self.lane_match,
                    )

                cls_pred = predictions[:, :2]
                if self.lane_iou_cls_on:
                    # S1.3 (FIXED): soft cls target encodes geometric quality but
                    # keeps a strong positive FLOOR. NB105-bug: using the raw
                    # matched line-IoU (~0.05 early) as the target drove EVERY
                    # positive score to ~0.05, collapsing score_std to 0.005 and
                    # zeroing F1 under the tau=0.4 threshold. Fix: map the IoU
                    # into [floor, 1] via floor + (1-floor)*(IoU/iou_norm), so a
                    # matched prior always targets >= floor (a clear positive vs
                    # the 0 negatives) while still RANKING positives by quality.
                    # floor=0.5, iou_norm=0.3 (typical good-match IoU at this
                    # stage) => IoU 0.05->0.58, 0.3->1.0. Negatives stay 0, so
                    # the distribution separates instead of collapsing.
                    with torch.no_grad():
                        _rp = predictions[matched_row_inds, 6:] * (img_w - 1)
                        _rt = target[matched_col_inds, 6:] * (img_w - 1)
                        _ious = _lane_match_line_iou(_rp, _rt, img_w).clamp(0, 1)
                        _floor, _norm = 0.5, 0.3
                        _q = (_floor + (1 - _floor) * (_ious / _norm)).clamp(_floor, 1.0)
                    soft_t = predictions.new_zeros(predictions.shape[0])
                    soft_t[matched_row_inds] = _q
                    cls_loss_sum = cls_loss_sum + (
                        _qfl_loss(cls_pred, soft_t, gamma=self.lane_iou_cls_gamma).sum()
                        / target.shape[0]
                    )
                else:
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_target[matched_row_inds] = 1
                    cls_loss_sum = cls_loss_sum + (
                        self.lane_cls_loss_fn(cls_pred, cls_target).sum()
                        / target.shape[0]
                    )

                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl = reg_yxtl.clone()
                reg_yxtl[:, 0] = reg_yxtl[:, 0] * n_strips
                reg_yxtl[:, 1] = reg_yxtl[:, 1] * (img_w - 1)
                reg_yxtl[:, 2] = reg_yxtl[:, 2] * 180
                reg_yxtl[:, 3] = reg_yxtl[:, 3] * n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()
                target_yxtl[:, 0] = target_yxtl[:, 0] * n_strips
                target_yxtl[:, 2] = target_yxtl[:, 2] * 180

                # P8 iter-1 fix: CLRHead assumes predictions[:, :, 2:5] live in
                # [0, 1] (the prior init range) and `length` lives in
                # [0, n_strips]. Only `nn.init.normal_(..., std=1e-3)` keeps
                # them there - works for CLRKDNet's lane-only setup, but in our
                # 35M-param joint RT-DETR + lane training the GCA adapter
                # output and shared backbone gradients let `reg[:, :, :4]` drift
                # to magnitude ~10+ on some priors. Multiplied by 71/639/180/71
                # this produced single-batch xytl_loss spikes of 6000+ that
                # poisoned the LossMonitor EMA. Clamp the per-element pixel-
                # space diff to +-100 so one outlier prior can contribute at
                # most 99.5 to smooth_l1 per element. In-range priors keep
                # full gradient (clamp's grad = 1 inside the range).
                # Diagnostic knob LANE_DIFF_CLAMP (env): the +-100 clamp
                # ZEROES the gradient for any matched prior whose start_x is
                # >100 px off target - which early in training is most of
                # them, so it may be a prime reason reg_layers never learns
                # the geometry. NB96 sets LANE_DIFF_CLAMP=none to test that.
                _clamp = self.xytl_diff_clamp
                raw_diff = reg_yxtl - target_yxtl
                diff = raw_diff if _clamp is None else raw_diff.clamp(min=-_clamp, max=_clamp)
                xytl_loss_sum = xytl_loss_sum + F.smooth_l1_loss(
                    diff, torch.zeros_like(diff), reduction='none',
                ).mean()

                # S1.4b: ASYMMETRIC "too-short" length hinge. reg_yxtl[:,3] and
                # target_yxtl[:,3] are both in STRIP units here (pred x n_strips;
                # GT is a strip count). The smooth-L1 above penalizes length
                # error SYMMETRICALLY and as 1/4 of a mixed-scale mean, so the
                # model freely SHORTENS lanes (NB107 length_mean 0.155->0.094) to
                # dodge x-error on hard rows. This term adds a one-sided penalty
                # on shortfall ONLY -- (gt_len - pred_len)_+, normalized to [0,1]
                # by n_strips so the weight is interpretable -- directly raising
                # the cost of predicting too-short lanes. Off by default (w=0).
                if self.lane_len_hinge_w > 0 and reg_yxtl.shape[0] > 0:
                    _ns = float(n_strips) if n_strips else 1.0
                    short_by = (target_yxtl[:, 3] - reg_yxtl[:, 3]).clamp(min=0.0) / _ns
                    xytl_loss_sum = xytl_loss_sum + self.lane_len_hinge_w * short_by.mean()

                reg_pred = predictions[matched_row_inds, 6:] * (img_w - 1)
                reg_targets = target[matched_col_inds, 6:] * (img_w - 1)
                # S1.4: y-reweight the per-row x error. 'near' = heavier on the
                # bottom (near-camera) rows where length shrinks; 'angle' =
                # normalize by local |dx/dy| so a fixed angular error costs the
                # same everywhere. Applied as a per-row weight on the IoU-loss
                # x-grid (rows run far->near top->bottom in the 78-D layout).
                if self.lane_y_reweight in ('near', 'angle') and reg_pred.shape[0] > 0:
                    iou_loss_sum = iou_loss_sum + _y_reweighted_liou(
                        reg_pred, reg_targets, img_w, self.lane_iou_loss_fn,
                        mode=self.lane_y_reweight)
                else:
                    iou_loss_sum = iou_loss_sum + self.lane_iou_loss_fn(
                        reg_pred, reg_targets, img_w, length=15,
                    )

                # S1.2: 2nd-difference (curvature) smoothness penalty on the
                # matched lane's x-offsets. OPTIONAL (S0.4 found near already
                # smoother than far) - only active when LANE_SMOOTH_W > 0.
                if self.lane_smooth_w > 0 and reg_pred.shape[0] > 0:
                    _xn = predictions[matched_row_inds, 6:]      # normalized x
                    _d2 = _xn[:, 2:] - 2 * _xn[:, 1:-1] + _xn[:, :-2]
                    xytl_loss_sum = xytl_loss_sum + self.lane_smooth_w * (_d2 ** 2).mean()

        denom = float(batch_size * refine_layers)
        cls_loss_avg = cls_loss_sum / denom
        xytl_loss_avg = xytl_loss_sum / denom
        iou_loss_avg = iou_loss_sum / denom

        # Aux seg: CLRHead's internal binary head outputs (B, num_classes_aux, H, W).
        # Our cfg has num_classes_aux=2 (bg + lane). seg_batch carries a
        # (B, 1, H, W) lane mask; downsample-resize to lane_seg's spatial.
        gt_lane_mask = seg_batch['lane_seg_mask'].to(lane_seg.device).float()
        if gt_lane_mask.dim() == 4 and gt_lane_mask.shape[1] == 1:
            gt_lane_mask = gt_lane_mask.squeeze(1)
        if lane_seg.shape[-2:] != gt_lane_mask.shape[-2:]:
            gt_lane_mask = F.interpolate(
                gt_lane_mask.unsqueeze(1).float(),
                size=lane_seg.shape[-2:],
                mode='nearest',
            ).squeeze(1)
        lane_seg_target = gt_lane_mask.long().clamp(0, 1)
        lane_seg_loss = F.cross_entropy(lane_seg, lane_seg_target, ignore_index=255)

        total_loss.update({
            'lane_cls_loss': cls_loss_avg * self.lane_cls_weight,
            'lane_xytl_loss': xytl_loss_avg * self.lane_xytl_weight,
            'lane_iou_loss': iou_loss_avg * self.lane_iou_weight,
            'lane_seg_aux_loss': lane_seg_loss * self.lane_seg_weight,
        })
        return total_loss

    def _compute_bezier_only_loss(self, lane_seg_output, seg_batch, total_loss):
        """B5: 4-term Bezier lane loss with optional complexity penalty.

        lane_seg_output shape from `LaneBezierHead.forward(training=True)`:
            {
                'lane_output': {'predictions_lists': [(B, P, 16), ...],
                                'seg': (B, 2, H_s, W_s)},
                'lcm_output': {'degree_weights': (B, P, 3), ...} | None,
            }
        seg_batch carries:
            'lane_targets'    (B, max_lanes, 16) - Bezier targets
            'lane_seg_mask'   (B, 1, H, W)       - aux supervision mask
        """
        import torch.nn.functional as F

        lane_output = lane_seg_output['lane_output']
        lcm_output = lane_seg_output.get('lcm_output', None)
        if not isinstance(lane_output, dict):
            # Eval mode - skip loss computation.
            total_loss.update({
                'lane_cls_loss': torch.tensor(0.0, device=self.device),
                'lane_xytl_loss': torch.tensor(0.0, device=self.device),
                'lane_iou_loss': torch.tensor(0.0, device=self.device),
                'lane_seg_aux_loss': torch.tensor(0.0, device=self.device),
                'lane_complexity_loss': torch.tensor(0.0, device=self.device),
            })
            return total_loss

        predictions_lists = lane_output['predictions_lists']
        lane_seg = lane_output['seg']
        lane_targets = seg_batch['lane_targets']  # (B, max_lanes, 16)
        img_w = img_h = 640

        device = predictions_lists[0].device
        batch_size = lane_targets.shape[0]
        refine_layers = len(predictions_lists)

        cls_loss_sum = torch.tensor(0.0, device=device)
        geom_loss_sum = torch.tensor(0.0, device=device)
        iou_loss_sum = torch.tensor(0.0, device=device)

        # LCM degree weights are shared across all stages (computed at the
        # final-stage features in our impl); index per matched prior.
        deg_weights_all = None
        if lcm_output is not None and 'degree_weights' in lcm_output:
            deg_weights_all = lcm_output['degree_weights']  # (B, P, 3)

        # Build a per-prior "matched" mask for the complexity penalty so
        # the penalty only counts positives (matched lanes).
        matched_mask_per_batch = []

        for stage in range(refine_layers):
            predictions_list = predictions_lists[stage]
            for b, (predictions, target) in enumerate(zip(predictions_list, lane_targets)):
                # Drop padding lanes (pos score == 0 in the target)
                target_pos = target[target[:, 1] == 1]

                if target_pos.shape[0] == 0:
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss_sum = cls_loss_sum + self.lane_cls_loss_fn(
                        cls_pred, cls_target,
                    ).sum()
                    if stage == refine_layers - 1:
                        matched_mask_per_batch.append(
                            torch.zeros(predictions.shape[0], dtype=torch.bool, device=device),
                        )
                    continue

                with torch.no_grad():
                    matched_row, matched_col = self.bezier_assign(
                        predictions, target_pos, img_w=img_w,
                        match=self.lane_match,
                    )

                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row] = 1
                cls_pred = predictions[:, :2]
                cls_loss_sum = cls_loss_sum + (
                    self.lane_cls_loss_fn(cls_pred, cls_target).sum()
                    / target_pos.shape[0]
                )

                if stage == refine_layers - 1:
                    mm = torch.zeros(predictions.shape[0], dtype=torch.bool, device=device)
                    mm[matched_row] = True
                    matched_mask_per_batch.append(mm)

                # Geometric loss (control points + validity range)
                pred_match = predictions[matched_row]
                targ_match = target_pos[matched_col]
                # Optional LCM mixture for the IoU side. Geom loss
                # operates on raw control points (it's the regression
                # target for the CPs themselves, not the rendered curve).
                deg_match = None
                if deg_weights_all is not None:
                    deg_match = deg_weights_all[b][matched_row]

                geom_loss_sum = geom_loss_sum + self.bezier_geom_loss_fn(
                    pred_match, targ_match,
                )
                iou_loss_sum = iou_loss_sum + self.bezier_iou_loss_fn(
                    pred_match, targ_match,
                    img_w=img_w, pred_degree_weights=deg_match,
                )

        denom = float(batch_size * refine_layers)
        cls_loss_avg = cls_loss_sum / denom
        geom_loss_avg = geom_loss_sum / denom
        iou_loss_avg = iou_loss_sum / denom

        # Aux seg
        gt_lane_mask = seg_batch['lane_seg_mask'].to(lane_seg.device).float()
        if gt_lane_mask.dim() == 4 and gt_lane_mask.shape[1] == 1:
            gt_lane_mask = gt_lane_mask.squeeze(1)
        if lane_seg.shape[-2:] != gt_lane_mask.shape[-2:]:
            gt_lane_mask = F.interpolate(
                gt_lane_mask.unsqueeze(1).float(),
                size=lane_seg.shape[-2:],
                mode='nearest',
            ).squeeze(1)
        lane_seg_target = gt_lane_mask.long().clamp(0, 1)
        lane_seg_loss = F.cross_entropy(lane_seg, lane_seg_target, ignore_index=255)

        total_loss.update({
            'lane_cls_loss': cls_loss_avg * self.lane_cls_weight,
            # Keep the legacy `lane_xytl_loss` key so the trainer's
            # ll_seg = sum-of-4 aggregation in tasks.py picks up the
            # geom loss without needing a separate aggregation tweak.
            'lane_xytl_loss': geom_loss_avg * self.bezier_geom_weight,
            'lane_iou_loss': iou_loss_avg * self.lane_iou_weight,
            'lane_seg_aux_loss': lane_seg_loss * self.lane_seg_weight,
        })

        # Complexity penalty (LCM only). When LCM is off, contribute 0.
        if deg_weights_all is not None and matched_mask_per_batch:
            matched_mask = torch.stack(matched_mask_per_batch, dim=0)  # (B, P)
            complexity_loss = self.complexity_loss_fn(
                deg_weights_all, matched_mask=matched_mask,
            )
            total_loss['lane_complexity_loss'] = (
                complexity_loss * self.complexity_weight
            )
        else:
            total_loss['lane_complexity_loss'] = torch.tensor(0.0, device=device)
        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
