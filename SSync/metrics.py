import math
from typing import Dict, Optional, Sequence, Tuple

import einops
import numpy as np
import torch
import torchmetrics

from SSync.utils import make_build_fn


@make_build_fn(__name__, "metric")
def build(config, name: str):
    pass  # No special module building needed


class Metric(torchmetrics.Metric):
    def __init__(self, input_mapping: Dict[str, str], **kwargs) -> None:
        super().__init__(**kwargs)
        # Mapping from parameter in _update to name in inputs dict
        self.input_mapping = input_mapping

    def update(self, *args, **kwargs):
        inputs = {}
        for mapped_key, input_key in self.input_mapping.items():
            if input_key is None:
                continue
            if input_key not in kwargs:
                raise ValueError(
                    f"Key {input_key} not found in inputs to metric. "
                    f"Available inputs are: {list(kwargs)}"
                )
            inputs[mapped_key] = kwargs[input_key]

        return self._update(*args, **inputs)

    def _update(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclasses")


class VideoIdentification(Metric):
    """Fragmentation metric for videos.

    Returns the average number of slots per GT object with IoU >= threshold.
    Lower is better (less over-fragmentation).

    Inputs:
        true_mask: (B, T, O, H, W)  binary GT object masks (objects can be empty)
        pred_mask: (B, T, S, H, W)  slot masks (binary or logits; will be thresholded at 0.5)
    """

    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        iou_threshold: float = 0.5,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        bgclass: bool=True,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.iou_threshold = float(iou_threshold)
        self.bgclass = bgclass
        # numerator: total #slots assigned across all valid GT objects
        self.add_state("slot_counts_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        # denominator: total #valid GT objects (area > 0)
        self.add_state("valid_obj_count", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    @torch.no_grad()
    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        # Shape checks: (B,T,O,H,W) and (B,T,S,H,W)
        _check_shape(true_mask, (None, None, None, None, None), "true_mask [B,T,O,H,W]")
        _check_shape(pred_mask, (true_mask.shape[0], true_mask.shape[1], None, true_mask.shape[3], true_mask.shape[4]),
                     "pred_mask [B,T,S,H,W]")

        # binarize if necessary
        if true_mask.dtype != torch.bool:
            true_mask = (true_mask > 0.5)
        if pred_mask.dtype != torch.bool:
            pred_mask = (pred_mask > 0.5)
        
        if self.bgclass: 
            true_mask = true_mask[:, :, 1:]
        B, T, O, H, W = true_mask.shape
        S = pred_mask.shape[2]

        # areas
        slot_area = pred_mask.sum(dim=(-1, -2))     # (B,T,S)
        obj_area  = true_mask.sum(dim=(-1, -2))     # (B,T,O)
        per_slot_cnt = obj_area.reshape(-1, O).float()

        valid_obj = obj_area > 0                    # (B,T,O)
        total_valid_objs = valid_obj.sum()

        if total_valid_objs == 0:
            return  # nothing to accumulate

        # Loop over object index to keep memory low; vectorize over B,T,S
        identified_obj_counts = 0.0
        valid_obj_counts = 0
        for o in range(O):
            gt_o = true_mask[:, :, o]                              # (B,T,H,W)
            valid_o = valid_obj[:, :, o]                           # (B,T)

            if not valid_o.any():
                continue

            inter = (pred_mask & gt_o.unsqueeze(2)).sum(dim=(-1, -2)).to(torch.float64)  # (B,T,S)
            union = (slot_area + obj_area[:, :, o].unsqueeze(-1) - inter).to(torch.float64)
            iou   = torch.where(union > 0, inter / union, torch.zeros_like(union, dtype=torch.float64)) # 

            # per-object slot counts (B,T)
            slot_cnt_for_o = (iou > self.iou_threshold).sum(dim=-1).to(torch.float64)
            # mask invalid objects
            slot_cnt_for_o = torch.where(valid_o, slot_cnt_for_o, torch.zeros_like(slot_cnt_for_o)) # (B,T)
            # valid_obj : (B,T) 
            
            used_obj_bool = valid_o & (slot_cnt_for_o > 0)        # (B,T) bool
            used_obj = used_obj_bool.to(slot_cnt_for_o.dtype)     # (B,T) {0,1} same dtype as slot_cnt_for_o

            # slot_counts_sum += slot_cnt_for_o.sum()
            identified_obj_counts += used_obj.sum()
            valid_obj_counts += valid_o.sum()

        self.slot_counts_sum += identified_obj_counts
        self.valid_obj_count += valid_obj_counts

    def compute(self):
        if self.valid_obj_count == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        return self.slot_counts_sum / self.valid_obj_count

class VideoFragmentation(Metric):
    """Fragmentation metric for videos.

    Returns the average number of slots per GT object with IoU >= threshold.
    Lower is better (less over-fragmentation).

    Inputs:
        true_mask: (B, T, O, H, W)  binary GT object masks (objects can be empty)
        pred_mask: (B, T, S, H, W)  slot masks (binary or logits; will be thresholded at 0.5)
    """

    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        iou_threshold: float = 0.5,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        bgclass: bool=True,
        onlycorrect: bool=False
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.iou_threshold = float(iou_threshold)
        self.bgclass = bgclass
        self.onlycorrect = onlycorrect
        # numerator: total #slots assigned across all valid GT objects
        self.add_state("slot_counts_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        # denominator: total #valid GT objects (area > 0)
        self.add_state("valid_obj_count", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    @torch.no_grad()
    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        # Shape checks: (B,T,O,H,W) and (B,T,S,H,W)
        _check_shape(true_mask, (None, None, None, None, None), "true_mask [B,T,O,H,W]")
        _check_shape(pred_mask, (true_mask.shape[0], true_mask.shape[1], None, true_mask.shape[3], true_mask.shape[4]),
                     "pred_mask [B,T,S,H,W]")

        # binarize if necessary
        if true_mask.dtype != torch.bool:
            true_mask = (true_mask > 0.5)
        if pred_mask.dtype != torch.bool:
            pred_mask = (pred_mask > 0.5)
        if self.bgclass: 
            true_mask = true_mask[:, :, 1:]
        B, T, O, H, W = true_mask.shape
        S = pred_mask.shape[2]

        # areas
        slot_area = pred_mask.sum(dim=(-1, -2))     # (B,T,S)
        obj_area  = true_mask.sum(dim=(-1, -2))     # (B,T,O)
        per_slot_cnt = obj_area.reshape(-1, O).float()

        valid_obj = obj_area > 0                    # (B,T,O)
        total_valid_objs = valid_obj.sum()

        if total_valid_objs == 0:
            return  # nothing to accumulate

        # Loop over object index to keep memory low; vectorize over B,T,S
        slot_counts_sum = 0.0
        valid_obj_counts = 0
        for o in range(O):
            gt_o = true_mask[:, :, o]                              # (B,T,H,W)
            valid_o = valid_obj[:, :, o]                           # (B,T)

            if not valid_o.any():
                continue

            inter = (pred_mask & gt_o.unsqueeze(2)).sum(dim=(-1, -2)).to(torch.float64)  # (B,T,S)
            union = (slot_area + obj_area[:, :, o].unsqueeze(-1) - inter).to(torch.float64)
            iou   = torch.where(union > 0, inter / union, torch.zeros_like(union, dtype=torch.float64)) # 

            # per-object slot counts (B,T)
            slot_cnt_for_o = (iou > self.iou_threshold).sum(dim=-1).to(torch.float64)
            # mask invalid objects
            slot_cnt_for_o = torch.where(valid_o, slot_cnt_for_o, torch.zeros_like(slot_cnt_for_o)) # (B,T)
            # valid_obj : (B,T) 
            used_obj_bool = valid_o & (slot_cnt_for_o > 0)        # (B,T) bool
            used_obj = used_obj_bool.to(slot_cnt_for_o.dtype)     # (B,T) {0,1} same dtype as slot_cnt_for_o

            slot_counts_sum += slot_cnt_for_o.sum()
            valid_obj_counts += used_obj.sum()

        self.slot_counts_sum += slot_counts_sum
        if self.onlycorrect:
            self.valid_obj_count += valid_obj_counts
        else:
            self.valid_obj_count += total_valid_objs

    def compute(self):
        if self.valid_obj_count == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        return self.slot_counts_sum / self.valid_obj_count

class VideoSlotIoUFragmentation(Metric):
    """Fragmentation metric for videos.

    Returns the average number of slots per GT object with IoU >= threshold.
    Lower is better (less over-fragmentation).

    Inputs:
        true_mask: (B, T, O, H, W)  binary GT object masks (objects can be empty)
        pred_mask: (B, T, S, H, W)  slot masks (binary or logits; will be thresholded at 0.5)
    """

    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        iou_threshold: float = 0.5,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        bgclass: bool=True,
        onlycorrect: bool=False
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.iou_threshold = float(iou_threshold)
        self.bgclass = bgclass
        self.onlycorrect = onlycorrect
        # numerator: total #slots assigned across all valid GT objects
        self.add_state("slot_counts_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        # denominator: total #valid GT objects (area > 0)
        self.add_state("valid_obj_count", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    @torch.no_grad()
    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        # Shape checks: (B,T,O,H,W) and (B,T,S,H,W)
        _check_shape(true_mask, (None, None, None, None, None), "true_mask [B,T,O,H,W]")
        _check_shape(pred_mask, (true_mask.shape[0], true_mask.shape[1], None, true_mask.shape[3], true_mask.shape[4]),
                     "pred_mask [B,T,S,H,W]")

        # binarize if necessary
        if true_mask.dtype != torch.bool:
            true_mask = (true_mask > 0.5)
        if pred_mask.dtype != torch.bool:
            pred_mask = (pred_mask > 0.5)
        
        if self.bgclass: 
            true_mask = true_mask[:, :, 1:]
        B, T, O, H, W = true_mask.shape
        S = pred_mask.shape[2]

        # areas
        slot_area = pred_mask.sum(dim=(-1, -2))     # (B,T,S)
        obj_area  = true_mask.sum(dim=(-1, -2))     # (B,T,O)
        per_slot_cnt = obj_area.reshape(-1, O).float()

        valid_obj = obj_area > 0                    # (B,T,O)
        total_valid_objs = valid_obj.sum()

        if total_valid_objs == 0:
            return  # nothing to accumulate

        # Loop over object index to keep memory low; vectorize over B,T,S
        slot_counts_sum = 0.0
        valid_obj_counts = 0
        for o in range(O):
            gt_o = true_mask[:, :, o]                              # (B,T,H,W)
            valid_o = valid_obj[:, :, o]                           # (B,T)

            if not valid_o.any():
                continue

            inter = (pred_mask & gt_o.unsqueeze(2)).sum(dim=(-1, -2)).to(torch.float64)  # (B,T,S)
            union = slot_area.to(torch.float64)
            # union = (slot_area + obj_area[:, :, o].unsqueeze(-1) - inter).to(torch.float64)
            iou   = torch.where(union > 0, inter / union, torch.zeros_like(union, dtype=torch.float64)) # 

            # per-object slot counts (B,T)
            slot_cnt_for_o = (iou > self.iou_threshold).sum(dim=-1).to(torch.float64)
            # mask invalid objects
            slot_cnt_for_o = torch.where(valid_o, slot_cnt_for_o, torch.zeros_like(slot_cnt_for_o)) # (B,T)
            # valid_obj : (B,T) 
            used_obj_bool = valid_o & (slot_cnt_for_o > 0)        # (B,T) bool
            used_obj = used_obj_bool.to(slot_cnt_for_o.dtype)     # (B,T) {0,1} same dtype as slot_cnt_for_o

            slot_counts_sum += slot_cnt_for_o.sum()
            valid_obj_counts += used_obj.sum()

        self.slot_counts_sum += slot_counts_sum
        if self.onlycorrect:
            self.valid_obj_count += valid_obj_counts
        else:
            self.valid_obj_count += total_valid_objs

    def compute(self):
        if self.valid_obj_count == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        return self.slot_counts_sum / self.valid_obj_count



class VideoUnusedSlotFraction(Metric):
    """Unused-slot fraction for videos.

    Returns the fraction of slots that do not cover foreground (background-only slots).

    Inputs:
        true_mask: (B, T, O, H, W)  binary GT object masks
        pred_mask: (B, T, S, H, W)  slot masks (binary or logits; will be thresholded at 0.5)

    Args:
        criterion:
            - "max_iou": a slot is 'used' if max_o IoU(slot, obj_o) >= τ
            - "fg_coverage": a slot is 'used' if (slot ∧ FG).sum / slot.sum >= τ  where FG = any object
    """

    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        iou_threshold: float = 0.1,
        criterion: str = "max_iou",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        if criterion not in ("max_iou", "fg_coverage"):
            raise ValueError("criterion must be 'max_iou' or 'fg_coverage'")
        self.iou_threshold = float(iou_threshold)
        self.criterion = criterion
        # accumulate counts to compute a global fraction
        self.add_state("unused_count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_slots",  default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    @torch.no_grad()
    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        _check_shape(true_mask, (None, None, None, None, None), "true_mask [B,T,O,H,W]")
        _check_shape(pred_mask, (true_mask.shape[0], true_mask.shape[1], None, true_mask.shape[3], true_mask.shape[4]),
                     "pred_mask [B,T,S,H,W]")

        if true_mask.dtype != torch.bool:
            true_mask = (true_mask > 0.5)
        if pred_mask.dtype != torch.bool:
            pred_mask = (pred_mask > 0.5)

        B, T, O, H, W = true_mask.shape
        S = pred_mask.shape[2]

        # total slot instances considered
        self.total_slots += torch.tensor(B * T * S, dtype=torch.int64, device=pred_mask.device)

        if self.criterion == "fg_coverage":
            fg = true_mask.any(dim=2)                                         # (B,T,H,W)
            inter = (pred_mask & fg.unsqueeze(2)).sum(dim=(-1, -2)).to(torch.float64)   # (B,T,S)
            area  = pred_mask.sum(dim=(-1, -2)).to(torch.float64).clamp_min(1.0)        # (B,T,S)
            cov   = inter / area
            unused = (cov < self.iou_threshold)
            self.unused_count += unused.sum().to(torch.int64)
            return

        # criterion == "max_iou"
        slot_area = pred_mask.sum(dim=(-1, -2)).to(torch.float64)             # (B,T,S)
        max_iou = torch.zeros((B, T, S), dtype=torch.float64, device=pred_mask.device)

        for o in range(O):
            gt_o = true_mask[:, :, o]                                         # (B,T,H,W)
            inter = (pred_mask & gt_o.unsqueeze(2)).sum(dim=(-1, -2)).to(torch.float64)  # (B,T,S)
            union = slot_area + gt_o.sum(dim=(-1, -2)).unsqueeze(-1).to(torch.float64) - inter
            iou_o = torch.where(union > 0, inter / union, torch.zeros_like(union))
            max_iou = torch.maximum(max_iou, iou_o)

        unused = (max_iou < self.iou_threshold)
        self.unused_count += unused.sum().to(torch.int64)

    def compute(self):
        if self.total_slots == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        # fraction in [0,1]
        return self.unused_count.to(torch.float64) / self.total_slots.to(torch.float64)



class ImageMaskMetricMixin:
    """Mixin class for mask-based per-image metrics.

    Handles shape checking and rearranging of inputs.

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for ARI computation.
        flatten_spatially: If true, flatten spatial dimensions into a single dimension.
        move_classes_last: If true, move classes to the last dimension.
    """

    def __init__(
        self,
        video_input: bool,
        *args,
        flatten_spatially: bool = True,
        move_classes_last: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_input = video_input
        self.rearrange_pattern = self._get_rearrange_pattern(
            video_input, flatten_spatially, move_classes_last
        )

    @staticmethod
    def _get_rearrange_pattern(video_input, flatten_spatially, move_classes_last) -> str:
        if video_input:
            # For video input, the temporal dimension is folded into the batch dimension, i.e.
            # all frames are treated independently from the downstream metrics.
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b t c h w -> (b t) (h w) c"
                else:
                    pattern = "b t c h w -> (b t) c (h w)"
            else:
                if move_classes_last:
                    pattern = "b t c h w -> (b t) h w c"
                else:
                    pattern = "b t c h w -> (b t) c h w"
        else:
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b c h w -> b (h w) c"
                else:
                    pattern = "b c h w -> b c (h w)"
            else:
                if move_classes_last:
                    pattern = "b c h w -> b h w c"
                else:
                    pattern = "b c h w -> b c h w"

        return pattern

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
                width).
            pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
                width).
        """
        if self.video_input:
            _check_shape(
                true_mask,
                (None, None, None, None, None),
                "true_mask [bs, n_frames, n_true_classes, h, w]",
            )
            b, t, _, h, w = true_mask.shape
            _check_shape(
                pred_mask,
                (b, t, None, h, w),
                "pred_mask [bs, n_frames, n_pred_classes, h, w]",
            )
        else:
            _check_shape(true_mask, (None, None, None, None), "true_mask [bs, n_true_classes, h, w]")
            b, _, h, w = true_mask.shape
            _check_shape(pred_mask, (b, None, h, w), "pred_mask [bs, n_pred_classes, h, w]")

        true_mask = einops.rearrange(true_mask, self.rearrange_pattern)
        pred_mask = einops.rearrange(pred_mask, self.rearrange_pattern)

        return super()._update(true_mask, pred_mask)


class VideoMaskMetricMixin:
    """Mixin class for mask-based per-video metrics.

    Handles shape checking and rearranging of inputs.

    Args:
        flatten_temporally: If true, flatten temporal dimensions into the spatial dimensions.
            In this case, frames are vertically concatenated.
        flatten_spatially: If true, flatten spatial dimensions into a single dimension.
        move_classes_last: If true, move classes to the last dimension.
    """

    def __init__(
        self,
        *args,
        flatten_temporally: bool = True,
        flatten_spatially: bool = True,
        move_classes_last: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rearrange_pattern = self._get_rearrange_pattern(
            flatten_temporally, flatten_spatially, move_classes_last
        )

    @staticmethod
    def _get_rearrange_pattern(
        flatten_temporally: bool, flatten_spatially: bool, move_classes_last: bool
    ) -> str:
        if flatten_temporally:
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b t c h w -> b (t h w) c"
                else:
                    pattern = "b t c h w -> b c (t h w)"
            else:
                # Temporal dimension is folded into the height dimension, i.e. all frames are
                # vertically concatenated.
                if move_classes_last:
                    pattern = "b t c h w -> b (t h) w c"
                else:
                    pattern = "b t c h w -> b c (t h) w"
        else:
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b t c h w -> b t (h w) c"
                else:
                    pattern = "b t c h w -> b t c (h w)"
            else:
                if move_classes_last:
                    pattern = "b t c h w -> b t h w c"
                else:
                    pattern = "b t c h w -> b t c h w"

        return pattern

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
            pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
                width).
        """
        _check_shape(
            true_mask,
            (None, None, None, None, None),
            "true_mask [bs, n_frames, n_true_classes, h, w]",
        )
        b, t, _, h, w = true_mask.shape
        _check_shape(
            pred_mask,
            (b, t, None, h, w),
            "pred_mask [bs, n_frames, n_pred_classes, h, w]",
        )
        true_mask = einops.rearrange(true_mask, self.rearrange_pattern)
        pred_mask = einops.rearrange(pred_mask, self.rearrange_pattern)

        return super()._update(true_mask, pred_mask)


class AdjustedRandIndex(Metric):
    """Abstract ARI metric."""

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_points, n_true_classes)
            pred_mask: One-hot predicted masks of shape (batch, n_points, n_pred_classes)
        """
        assert true_mask.ndim == 3
        assert pred_mask.ndim == 3
        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=-1) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=-1)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_background and torch.any(n_true_classes_per_point != 1.0):
            raise ValueError("`true_mask` is not one-hot")

        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            true_mask[overlaps] = 0.0  # ARI ignores pixels where all ground truth clusters are zero

        if self.ignore_background:
            true_mask = true_mask[..., 1:]  # Remove the background mask

        values = adjusted_rand_index(true_mask, pred_mask)

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes_per_point.sum(dim=-1) > 0
        values = values[non_empty]

        self.values += values.sum()
        self.total += len(values)

    def compute(self):
        return self.values / self.total


class ImageARI(ImageMaskMetricMixin, AdjustedRandIndex):
    """ARI metric for images.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
            width).
        pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
            width).

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for metric computation.
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        video_input: bool = False,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(video_input, ignore_background, ignore_overlaps, pred_key, true_key)


class VideoARI(VideoMaskMetricMixin, AdjustedRandIndex):
    """ARI metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(ignore_background, ignore_overlaps, pred_key, true_key)


def adjusted_rand_index(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> torch.Tensor:
    """Computes the adjusted Rand index (ARI), a clustering similarity score.

    Adapted to Pytorch from SAVi Jax implementation:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_clusters). The true cluster
            assignment encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_clusters). The predicted
            cluster assignment encoded as one-hot.

    Returns:
        ARI scores as a tensor of shape (batch_size,).
    """
    N = torch.einsum("bpc, bpk -> bck", true_mask.to(torch.float64), pred_mask.to(torch.float64))
    A = torch.sum(N, axis=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, axis=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, axis=1)

    rindex = torch.sum(N * (N - 1), axis=[1, 2])
    aindex = torch.sum(A * (A - 1), axis=1)
    bindex = torch.sum(B * (B - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clip(num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0.0, ari, 1.0)


class IntersectionOverUnion(Metric):
    """Abstract IoU metric."""

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        matching: str = "none",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.matching = matching
        if matching not in ("none", "overlap", "hungarian"):
            raise ValueError("`matching` needs to be 'none' or 'overlap' or 'hungarian'")
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_points, n_true_classes)
            pred_mask: One-hot predicted masks of shape (batch, n_points, n_pred_classes)
        """
        assert true_mask.ndim == 3
        assert pred_mask.ndim == 3
        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=-1) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=-1)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_background and torch.any(n_true_classes_per_point != 1.0):
            raise ValueError("`true_mask` is not one-hot")
        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            true_mask[overlaps] = 0.0
            pred_mask = pred_mask.clone()
            pred_mask[overlaps] = 0.0

        if self.ignore_background:
            true_mask = true_mask[..., 1:]  # Remove the background mask

        values = intersection_over_union_with_matching(
            true_mask, pred_mask, self.matching, empty_value=0.0
        )
        active_true_classes = true_mask.sum(dim=1) > 0

        # Compute mean IoU, ignoring empty true classes. This assumes that true-pred class pairs
        # with union==0 have been assigned zero IoU.
        n_true_classes = active_true_classes.sum(dim=-1)
        mean_iou = values.sum(dim=-1) / n_true_classes

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes > 0
        mean_iou = mean_iou[non_empty]

        self.values += mean_iou.sum()
        self.total += len(mean_iou)

    def compute(self):
        return self.values / self.total


class ImageIoU(ImageMaskMetricMixin, IntersectionOverUnion):
    """IoU metric for images.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
            width).
        pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
            width).

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for metric computation.
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
        matching: How to match true classes to predicted classes. For "none", assume classes are
            ordered, i.e. the true class at index i corresponds to the predicted class at index i.
            For "overlap", match the predicted class with the highest IoU to each true class.
    """

    def __init__(
        self,
        video_input: bool = False,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        matching: str = "none",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(
            video_input, ignore_background, ignore_overlaps, matching, pred_key, true_key
        )


class VideoIoU(VideoMaskMetricMixin, IntersectionOverUnion):
    """IoU metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
        matching: How to match true classes to predicted classes. For "none", assume classes are
            ordered, i.e. the true class at index i corresponds to the predicted class at index i.
            For "overlap", match the predicted class with the highest IoU to each true class.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        matching: str = "none",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(ignore_background, ignore_overlaps, matching, pred_key, true_key)


def intersection_over_union_with_matching(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    matching: str = "none",
    empty_value: float = 0.0,
) -> torch.Tensor:
    """Computes mask intersection-over-union, matching predicted masks to ground truth masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.
        matching: How to match true classes to predicted classes. For "none", assume classes are
            ordered, i.e. the true class at index i corresponds to the predicted class at index i.
            For "overlap", match the predicted class with the highest IoU to each true class.
        empty_value: Value to assume for the case when a class does not occur in the ground truth,
            and was also not predicted.

    Returns:
        IoU scores as a tensor of shape (batch_size, n_true_classes).
    """
    assert matching in ("none", "overlap", "hungarian")

    pairwise_ious = intersection_over_union(true_mask, pred_mask, empty_value)

    if matching == "none":
        if pairwise_ious.shape[1] != pairwise_ious.shape[2]:
            raise ValueError(
                "For matching 'none', n_true_classes needs to equal n_pred_classes, but is "
                f"{pairwise_ious.shape[1]} vs {pairwise_ious.shape[2]}"
            )
        ious = torch.diagonal(pairwise_ious, dim1=-2, dim2=-1)
    elif matching == "overlap":
        ious = torch.max(pairwise_ious, dim=2).values
    elif matching == "hungarian":
        all_true_idxs, all_pred_idxs = hungarian_matching(pairwise_ious, maximize=True)
        ious = torch.zeros(
            true_mask.shape[0], true_mask.shape[2], dtype=torch.float64, device=pairwise_ious.device
        )
        for idx, (true_idxs, pred_idxs) in enumerate(zip(all_true_idxs, all_pred_idxs)):
            ious[idx, true_idxs] = pairwise_ious[idx, true_idxs, pred_idxs]
    else:
        raise ValueError(f"Unknown matching for IoU `{matching}`")

    assert ious.shape == (true_mask.shape[0], true_mask.shape[2])
    return ious


def intersection_over_union(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    empty_value: float = 0.0,
):
    """Compute pairwise intersection-over-union between predicted and ground truth masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.
        empty_value: Value to assume for the case when a class does not occur in the ground truth,
            and was also not predicted.

    Returns:
        Pairwise IoU scores as a tensor of shape (batch_size, n_true_classes, n_pred_classes)."""
    intersection, fp, fn = confusion_matrix(true_mask, pred_mask)  # all B x C x K
    union = intersection + fp + fn
    ious = intersection / union

    # Deal with NaN from divide-by-zero (class does not occur and was not predicted)
    ious[union == 0] = empty_value

    return ious


def confusion_matrix(true_mask: torch.Tensor, pred_mask: torch.Tensor):
    """Computes confusion matrix between two sets of masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.

    Returns:
        Tuple containing the pairwise true positives (intersection), false positives and false
        negatives, all of shape (batch_size, n_true_classes, n_pred_classes).
    """
    true_mask = true_mask.to(torch.float64)
    pred_mask = pred_mask.to(torch.float64)

    true_positives = torch.einsum("bpc, bpk -> bck", true_mask, pred_mask)  # B x C x K
    n_true_points = true_mask.sum(1)  # B x C
    n_pred_points = pred_mask.sum(1)  # B x K

    false_positives = n_pred_points.unsqueeze(1) - true_positives  # B x C x K
    false_negatives = n_true_points.unsqueeze(2) - true_positives  # B x C x K

    return true_positives, false_positives, false_negatives


class JandFMetric(Metric):
    """Abstract Jaccard and F-score metric used in video object discovery.

    See also official implementation from DAVIS challenge:
    https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py
    """

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        metric_for_matching: str = "j_and_f",
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        if metric_for_matching not in ("j_and_f", "jaccard", "f_measure"):
            raise ValueError(
                "Matching should be one of 'j_and_f', 'jaccard', 'f_measure', but is "
                f"{metric_for_matching}"
            )
        self.metric_for_matching = metric_for_matching
        self.add_state(
            "j_and_f", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "jaccard", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "f_measure", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _aggregate_from_pairwise_values(pairwise_values, true_idxs, pred_idxs, n_true_classes):
        """

        Args:
            values: tensor of shape (batch, n_true_classes, n_pred_classes)
            true_idxs: tensor of shape (batch, n_true_classes)
            pred_idxs: tensor of shape (batch, n_pred_classes)
            n_true_classes: tensor of shape (batch,)

        Returns:
            Tensor of shape (batch_filtered,) where batch_filtered is batch size minus the number
            of true masks without any active mask, containing the mean metric value per sample.
        """
        values = torch.zeros(
            pairwise_values.shape[:2], dtype=torch.float64, device=pairwise_values.device
        )
        for idx, (t, p) in enumerate(zip(true_idxs, pred_idxs)):
            values[idx, t] = pairwise_values[idx, t, p]

        # Compute mean values, ignoring empty true masks. This assumes that empty masks have been
        # assigned a value of zero.
        values = values.sum(dim=-1) / n_true_classes

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes > 0
        values = values[non_empty]

        return values

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, [n_frames,] n_true_classes, height, width)
            pred_mask: One-hot predicted masks of shape (batch, [n_frames,] n_pred_classes, height,
                width)
        """
        assert true_mask.ndim == pred_mask.ndim == 4 or true_mask.ndim == pred_mask.ndim == 5
        batch_size = len(true_mask)
        if true_mask.ndim == 4:
            # Insert dummy video dimension if we are passed images
            true_mask = true_mask.unsqueeze(1)
            pred_mask = pred_mask.unsqueeze(1)

        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=2) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=2)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_background and torch.any(n_true_classes_per_point != 1.0):
            raise ValueError("`true_mask` is not one-hot")
        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            m = einops.repeat(overlaps, "b t h w -> b t c h w", c=true_mask.shape[2])
            true_mask[m] = 0.0
            pred_mask = pred_mask.clone()
            m = einops.repeat(overlaps, "b t h w -> b t k h w", k=pred_mask.shape[2])
            pred_mask[m] = 0.0

        if self.ignore_background:
            true_mask = true_mask[:, :, 1:]  # Remove the background mask

        # Jaccard is computed frame-by-frame in the original J & F implementation.
        all_jaccard = intersection_over_union(
            einops.rearrange(true_mask, "b t c h w -> (b t) (h w) c"),
            einops.rearrange(pred_mask, "b t k h w -> (b t) (h w) k"),
            empty_value=0.0,
        )  # B x true_classes x pred_classes
        all_jaccard = einops.rearrange(all_jaccard, "(b t) c k -> b t c k", b=batch_size)
        all_jaccard = all_jaccard.mean(1)

        # Boundary f-measure is computed frame-by-frame
        all_f_measure, _, _ = boundary_f_measure(
            einops.rearrange(true_mask, "b t c h w -> (b t) c h w"),
            einops.rearrange(pred_mask, "b t k h w -> (b t) k h w"),
        )  # (B * frames) x true_classes x pred_classes
        all_f_measure = einops.rearrange(all_f_measure, "(b t) c k -> b t c k", b=batch_size)
        all_f_measure = all_f_measure.mean(1)

        all_j_and_f = (all_jaccard + all_f_measure) / 2

        if self.metric_for_matching == "j_and_f":
            true_idxs, pred_idxs = hungarian_matching(all_j_and_f, maximize=True)
        elif self.metric_for_matching == "jaccard":
            true_idxs, pred_idxs = hungarian_matching(all_jaccard, maximize=True)
        elif self.metric_for_matching == "f_measure":
            true_idxs, pred_idxs = hungarian_matching(all_f_measure, maximize=True)
        else:
            raise ValueError(f"Unknown matching {self.metric_for_matching}")

        active_true_classes = true_mask.sum(dim=(1, -2, -1)) > 0  # B x true_classes
        n_true_classes = active_true_classes.sum(dim=-1)

        j_and_f = self._aggregate_from_pairwise_values(
            all_j_and_f, true_idxs, pred_idxs, n_true_classes
        )
        jaccard = self._aggregate_from_pairwise_values(
            all_jaccard, true_idxs, pred_idxs, n_true_classes
        )
        f_measure = self._aggregate_from_pairwise_values(
            all_f_measure, true_idxs, pred_idxs, n_true_classes
        )

        self.j_and_f += j_and_f.sum()
        self.jaccard += jaccard.sum()
        self.f_measure += f_measure.sum()
        self.total += len(j_and_f)

    def compute(self):
        return {
            "j_and_f": self.j_and_f / self.total,
            "jaccard": self.jaccard / self.total,
            "boundary_f_measure": self.f_measure / self.total,
        }


class ImageJandF(ImageMaskMetricMixin, JandFMetric):
    """J&F metric for images.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
            width).
        pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
            width).

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for metric computation.
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        video_input: bool = False,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        metric_for_matching: str = "j_and_f",
    ):
        super().__init__(
            video_input,
            ignore_background,
            ignore_overlaps,
            pred_key,
            true_key,
            flatten_spatially=False,
            move_classes_last=False,
            metric_for_matching=metric_for_matching,
        )


class VideoJandF(VideoMaskMetricMixin, JandFMetric):
    """J&F metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        metric_for_matching: str = "j_and_f",
    ):
        super().__init__(
            ignore_background,
            ignore_overlaps,
            pred_key,
            true_key,
            flatten_temporally=False,
            flatten_spatially=False,
            move_classes_last=False,
            metric_for_matching=metric_for_matching,
        )

class VideoJ(VideoJandF):
    """Returns mean IoU (J) from the VideoJandF metric."""

    def compute(self):
        scores = super().compute()
        return scores["jaccard"]


class VideoF(VideoJandF):
    """Returns boundary F-score from the VideoJandF metric."""

    def compute(self):
        scores = super().compute()
        return scores["boundary_f_measure"]
    
def boundary_f_measure(true_mask: torch.Tensor, pred_mask: torch.Tensor):
    """Computes pairwise F-measure on boundaries of masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_true_classes, height, width). The true
            masks encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_pred_classes, height, width). The
            predicted masks encoded as one-hot with missing values allowed.

    Returns:
        Tuple containing the pairwise f-measure, precision and recall, all of shape (batch_size,
        n_true_classes, n_pred_classes).
    """
    true_boundaries = einops.rearrange(masks_to_boundaries(true_mask), "b c h w -> b (h w) c")
    pred_boundaries = einops.rearrange(masks_to_boundaries(pred_mask), "b c h w -> b (h w) c")
    return f_measure(true_boundaries, pred_boundaries)


def f_measure(true_mask: torch.Tensor, pred_mask: torch.Tensor, empty_value: float = 0.0):
    """Compute pairwise F-measure between predicted and ground truth masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.
        empty_value: Value to assume for the case when a class does not occur in the ground truth,
            or in the predicted classes.

    Returns:
        Tuple containing the pairwise f-measure, precision and recall, all of shape (batch_size,
        n_true_classes, n_pred_classes).
    """
    tp, fp, fn = confusion_matrix(true_mask, pred_mask)  # all B x C x K
    n_pred_points = tp + fp
    n_true_points = tp + fn

    precision = tp / n_pred_points
    recall = tp / n_true_points

    # Guard against NaNs
    precision[n_pred_points == 0] = empty_value
    recall[n_true_points == 0] = empty_value

    f_measure = 2 * precision * recall / (precision + recall)
    f_measure[precision + recall == 0] = 0

    return f_measure, precision, recall


def masks_to_boundaries(masks: torch.Tensor, dilation_ratio: float = 0.02) -> torch.Tensor:
    """Compute the boundaries around the provided masks using morphological operations.

    Returns a tensor of the same shape as the input masks containing the boundaries of each mask.

    This implementations is adapted from a non-merged PR to torchvision, see
    https://github.com/pytorch/vision/pull/7704/.

    Args:
        masks: masks to transform of shape ([batch_size,], n_masks, height, width).
        dilation_ratio: ratio used for the dilation operation.

    Returns:
        Tensor of shape ([batch_size,], n_masks, height, width) with boundaries.
    """
    # If no masks are provided, return an empty tensor
    if masks.numel() == 0:
        return torch.zeros_like(masks)

    orig_dtype = masks.dtype
    if masks.ndim == 4:
        # Flatten batch dim into mask dim
        batch_size, n_masks = masks.shape[:2]
        masks = masks.flatten(0, 1)
    else:
        batch_size = None
    _, h, w = masks.shape
    img_diag = math.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    selem_size = dilation * 2 + 1
    selem = torch.ones((1, 1, selem_size, selem_size), device=masks.device)

    # Compute the boundaries for each mask
    masks = masks.float().unsqueeze(1)
    # The original erosion operation is emulated in two steps:
    #   1) Convolution counting the number of black pixels around each pixel
    #   2) Zero out pixel with less than the full number of black pixels
    # Step 2) effectively performs the minimum operation in erosion.
    eroded_masks = torch.nn.functional.conv2d(masks, selem, padding=dilation)
    eroded_masks = (eroded_masks == selem.sum()).byte()

    contours = masks.byte() - eroded_masks
    contours = contours.squeeze(1)

    if batch_size is not None:
        contours = contours.unflatten(0, (batch_size, n_masks))

    return contours.to(orig_dtype)


def hungarian_matching(
    cost_matrix: torch.Tensor, maximize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bipartite graph matching using Hungarian method.

    Naively computed over batches with a loop.

    Args:
        cost_matrix: Tensor of shape (batch_size, n_rows, n_cols) containing costs.

    Returns:
        Tuple of tensors of shape (batch_size, min_rows_cols, min_rows_cols) where min_rows_cols =
        min(n_rows, n_cols), with the first entry containing the matched row indices, and the
        second the matched column indices.
    """
    import scipy

    cost_matrix_cpu = cost_matrix.detach().cpu()

    rows, cols = [], []
    for elem in cost_matrix_cpu:
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(elem, maximize=maximize)
        rows.append(row_idxs)
        cols.append(col_idxs)

    row_idxs = torch.as_tensor(np.array(rows), dtype=torch.int64, device=cost_matrix.device)
    col_idxs = torch.as_tensor(np.array(cols), dtype=torch.int64, device=cost_matrix.device)

    return row_idxs, col_idxs


def _check_shape(x: torch.Tensor, expected_shape: Sequence[Optional[int]], name: str):
    """Verify shape of x is as expected.

    Adapted from SAVi Jax implementation:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py
    """
    if not isinstance(expected_shape, (list, tuple)):
        raise ValueError(
            "`expected_shape` should be a list or tuple of ints but got " f"{expected_shape}."
        )

    # Scalars have shape () by definition.
    shape = getattr(x, "shape", ())

    if len(shape) != len(expected_shape) or any(
        j is not None and i != j for i, j in zip(shape, expected_shape)
    ):
        raise ValueError(f"Input {name} has shape {shape}, but expected {expected_shape}.")
