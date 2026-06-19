from typing import Any, Dict, Optional, Tuple

import einops
import torch
from torch import nn

from SSync import modules, utils


@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        # self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # if self.positions_last:
        # prediction = prediction.transpose(-2, -1)
        # target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)

class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, temperature: float = 1.0, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)
        self.T = temperature
    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()
        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        prediction = torch.clamp(prediction / self.T, min=1e-9)
        
        return prediction

class MSECLSLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str,
                 temperature: float = 1.0, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)
        self.T = temperature

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach().long()  

        pred = utils.read_path(outputs, elements=self.pred_path)
        C = pred.shape[1]

        target_oh = torch.nn.functional.one_hot(target, num_classes=C)
        if target_oh.dim() > 2:
            d = target_oh.dim()
            # (B, ..., C) -> (B, C, ...)
            target_oh = target_oh.permute(0, d - 1, *range(1, d - 1))

        target_oh = target_oh.to(dtype=pred.dtype, device=pred.device)
        return target_oh

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        logits = utils.read_path(outputs, elements=self.pred_path)
        logits = logits / self.T
        probs = torch.softmax(logits, dim=1)  # (B, C, …)
        return probs




class NLLLoss(TorchLoss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        label_smoothing: float = 0.0,
        num_label: int = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, loss="NLLLoss", **kwargs)
        self.label_smoothing = label_smoothing
        self.num_label = num_label

        self.loss_fn = nn.NLLLoss()

        if self.label_smoothing > 0.0:
            assert self.num_label is not None, "num_label must be set when using label_smoothing > 0."
            self.target_prob = (
                self.label_smoothing / self.num_label + (1.0 - self.label_smoothing)
            )
        else:
            self.target_prob = None

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)
        target = target.detach()
        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        
        prediction = utils.read_path(outputs, elements=self.pred_path)
        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        
        if prediction.numel() == 0 or target.numel() == 0:
            return prediction.new_tensor(0.0, requires_grad=True)


        if self.target_prob is not None:
            with torch.no_grad():
                # prediction: [B, C], target: [B]
                p_true = prediction.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
                mask = p_true < self.target_prob

            if mask.any():
                prediction = prediction[mask]
                target = target[mask]
            else:
                return prediction.new_tensor(0.0, requires_grad=True)

        log_pred = prediction.clamp_min(1e-12).log()  
        return self.loss_fn(log_pred, target)




class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast

    def forward(self, slots, _):
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD
        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, S, D = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        loss = self.criterion(ss, target)
        return loss
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        features = features.squeeze(0)
        T, B, D = features.size() 

        ########################################################################
        mask = torch.kron(
            torch.eye(B, dtype=torch.bool, device=device),  # [batch, batch]
            torch.ones((T, T), dtype=torch.bool, device=device)  # [frames, frames]
        )  # → [batch*frames, batch*frames] boolean mask
        mask_size = mask.size(0)

        ignore = mask.clone()
        for i in range(B):
            start = i * T
            idx = torch.arange(start, start + T - 1, device=device)
            next_idx = idx + 1
            ignore[idx, next_idx] = False
            ignore[next_idx, idx] = False
        ignore = ~ignore
        ########################################################################
        mask = mask.float()
        ignore = ignore.float()

        mask = mask * ignore

        features_flat = features.permute(1, 0, 2).reshape(B * T, D)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features_flat, features_flat.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(mask_size, device=device).view(-1, 1), 0
        )
        mask = mask * logits_mask
        ignore_mask = ignore*logits_mask

        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.exp(logits) * ignore_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(mask_size).mean()

        return loss
class Img_Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast
        self.supcon = SupConLoss(temperature=temperature)

    def forward(self, slots, _):
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD

        loss = self.supcon(slots)

        return loss