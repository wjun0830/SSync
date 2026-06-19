from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchvision.utils import make_grid

from SSync import configuration, losses, modules, optimizers, utils, visualizations
from SSync.data.transforms import Denormalize
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)

    initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.target_encoder:
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
        assert (
            model_config.target_encoder_input is not None
        ), "Please specify `target_encoder_input`."

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video": # default input type
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder: # not in use
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None: # TransformerEncoder
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        if model_config.latent_processor:
            processor = modules.build_video(
                model_config.latent_processor,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = ["decoder", "grouping"]

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        loss_weights=model_config.get("loss_weights", None),
        target_encoder=target_encoder,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", True),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 25000),
        masks_to_visualize=masks_to_visualize,
        dif_n_bd=model_config.get("dif_n_bd", 1),
        dif_n_nbd=model_config.get("dif_n_nbd", 1),
        dif_n_bd_noise=model_config.get("dif_n_bd_noise", 1),
        merge_iou_threshold=model_config.get("merge_iou_threshold", 0.6),
        synergistic_warmup_ratio=model_config.get("synergistic_warmup_ratio", 0.3),
        max_steps=model_config.get("max_steps", 100000),
        experiment_group=model_config.get("experiment_group", "default_group"),
        experiment_name=model_config.get("experiment_name", "default_name"),
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        target_encoder: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        target_encoder_input: Optional[str] = None,
        visualize: bool = True,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, List[str]] = ["decoder", "grouping"],
        dif_n_bd: int = 1,
        dif_n_nbd: int = 1,
        dif_n_bd_noise: int = 1,
        merge_iou_threshold: float = 0.6,
        synergistic_warmup_ratio: float = 0.3,
        max_steps: int = 100000,
        experiment_group: str = "default_group",
        experiment_name: str = "default_name",
    ):
        super().__init__()
        self.dif_n_bd = dif_n_bd
        self.dif_n_nbd = dif_n_nbd
        self.dif_n_bd_noise = dif_n_bd_noise
        self.merge_iou_threshold = merge_iou_threshold
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor

        self.decoder = decoder
        self.target_encoder = target_encoder
        self.experiment_group = experiment_group
        self.experiment_name = experiment_name
        self.synergistic_warmup_ratio = synergistic_warmup_ratio
        self.max_steps = max_steps

        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
            loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            loss_weights_filtered = {
                k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            }
            self.loss_fns = nn.ModuleDict(loss_fns_filtered)
            self.loss_weights = loss_weights_filtered
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )
        self.vis_cnt = 0

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        return self.optimizer_builder(modules)

    def forward(self, inputs: Dict[str, Any], train=True) -> Dict[str, Any]:
        ### Save the intermediate checkpoints ###
        # if self.global_rank == 0 and train:
        #     if (self.trainer.global_step % (self.max_steps / 10) == 0) and (self.trainer.global_step > (self.max_steps / 10 * 2.9)):
        #         checkpoint_dir = "../logs/_" + self.experiment_name + "/checkpoints"
        #         prefix = self.experiment_group + "_step=step="
        #         pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.ckpt$")
        #         max_num = -1
        #         max_file = None
        #         for fname in os.listdir(checkpoint_dir):
        #             m = pattern.match(fname)
        #             if m:
        #                 num = int(m.group(1))
        #                 if num > max_num:
        #                     max_num, max_file = num, fname
        #         if max_file is None:
        #             raise FileNotFoundError("No matching checkpoint files found")
        #         src_path = os.path.join(checkpoint_dir, max_file)
        #         copy_step = max_num # self.trainer.global_step - 1
        #         dest_fname = f"{self.experiment_group}_copy={copy_step}.ckpt"
        #         dest_path = os.path.join(checkpoint_dir, dest_fname)
        #         shutil.copy(src_path, dest_path)
        #         print(f"Copied {max_file} → {dest_fname}")
        
        dif_n_bd = self.dif_n_bd
        dif_n_nbd = self.dif_n_nbd
        dif_n_bd_noise = self.dif_n_bd_noise
        
        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        ##### Forward pass #####
        ### Feature encoding 
        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]
        features_device = features.device
        B, T, HW, D = features.size()
        H = W = int(HW**0.5) if HW > 0 else 1

        ### Slot Attention
        slots_initial = self.initializer(batch_size=batch_size)
        processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]
        
        ### Slot Decoding -> Reconstruction
        decoder_output = self.decoder(slots)
        ##### Forward pass end #####

        ### Slot Attention Map
        grp_mask = processor_output["nosoftmax_dots"] 
        grp_mask_sm = processor_output["state_attn_mask"] 
        
        ### Decoder Object Map
        dec_mask = decoder_output["alpha"] # B T S HW
        dec_mask_sm = decoder_output["masks"] # B T S HW
             
        grp_labels = torch.argmax(grp_mask_sm, dim=2).reshape(B, T, H, W) # B T HW
        grp_mask_hw = grp_mask.permute(0, 1, 3, 2).reshape(B, T, H, W, -1)  # B T HW S
        
        dec_mask_hard = None
        if dec_mask_sm is not None:
            _, dec_mask_hard, _ = self.process_masks(
                dec_mask_sm, inputs, self.mask_resizers.get("decoder")
            )
        dec_mask_for_pred = dec_mask_hard if dec_mask_hard is not None else dec_mask # B T S HW
        dec_labels = torch.argmax(dec_mask_sm, dim=2).reshape(B, T, H, W) # B T HW
        dec_labels_for_eval = None
        if dec_mask_for_pred is not None:
            dec_labels_for_eval = torch.argmax(dec_mask_for_pred, dim=2)
            
            if dec_labels_for_eval.shape[-2:] != (H, W):
                dec_labels_for_eval = (
                    F.interpolate(
                        dec_labels_for_eval.view(-1, 1, dec_labels_for_eval.shape[-2], dec_labels_for_eval.shape[-1]).float(),
                        size=(H, W),
                        mode="nearest",
                    ).view(B, T, H, W).long()
                )
            dec_labels_for_eval = dec_labels_for_eval.reshape(B, T, H, W)
        dec_mask_hw = dec_mask.permute(0, 1, 3, 2).reshape(B, T, H, W, -1) # B T HW S

        ### --- [Just for Visualization] Detect Original boundaries & non-boundaries before merging --- ###
        ori_dec_padded_labels = torch.nn.functional.pad(dec_labels.float(), (1, 1, 1, 1, 1, 1), mode='replicate').long()
        ori_grp_padded_labels = torch.nn.functional.pad(grp_labels.float(), (1, 1, 1, 1, 1, 1), mode='replicate').long()
        ori_dec_neighbors = self.get_neighbor_patches(ori_dec_padded_labels)
        ori_grp_neighbors = self.get_neighbor_patches(ori_grp_padded_labels)

        ori_diff_neighbors_count = torch.zeros_like(dec_labels) # B T H W
        ori_grp_diff_neighbors_count = torch.zeros_like(grp_labels)
        for ori_grp_neighbor, ori_dec_neighbor in zip(ori_grp_neighbors, ori_dec_neighbors):
            ori_diff_neighbors_count += (dec_labels != ori_dec_neighbor)
            ori_grp_diff_neighbors_count += (grp_labels != ori_grp_neighbor)
            
        ori_is_boundary = ori_grp_diff_neighbors_count >= dif_n_bd
        ori_is_not_noise = (len(ori_grp_neighbors) - ori_grp_diff_neighbors_count) >= dif_n_bd_noise
        ori_is_boundary = ori_is_boundary & ori_is_not_noise
        ori_is_not_boundary = ori_diff_neighbors_count < dif_n_nbd


        ### --- Pseudo-Label Merging --- ###
        # Combine time and spatial dims before averaging per slot.
        per_slot_attn_mean = grp_mask.transpose(1, 2).flatten(2, 3).mean(-1) # B S
        per_slot_grp_mask_coverage = grp_mask > per_slot_attn_mean.unsqueeze(1).unsqueeze(-1) # B T S HW
        S = per_slot_grp_mask_coverage.shape[2]
        self.slot_num = S
        # compare slot coverage & identify redundant slots
        ### 1. compute iou between slots
        ### 2. identify redundant slots with iou > threshold
        ### 3. idneitfy the dominant slot for each redundant slot 
        ### 4. change the label of non-dominant slot to dominant slot if they are redundant
        # Compute IoU per frame (H*W) then average across time to keep temporal consistency.
        slot_masks_flat = per_slot_grp_mask_coverage.reshape(B, T, S, -1).float()  # B x T x S x (H*W)
        slot_areas = slot_masks_flat.sum(-1) + 1e-6  # B x T x S
        # Intersection via per-frame slot-wise matmul: (S x HW) @ (HW x S) -> S x S
        slot_masks_bt = slot_masks_flat.reshape(B * T, S, -1)
        intersection = torch.bmm(slot_masks_bt, slot_masks_bt.transpose(1, 2)).reshape(B, T, S, S)
        union = slot_areas.unsqueeze(-1) + slot_areas.unsqueeze(-2) - intersection
        slot_iou_t = torch.where(union > 0, intersection / union, torch.zeros_like(union))  # B x T x S x S
        slot_iou = slot_iou_t.mean(dim=1)  # B x S x S

        # Merge redundant slots within each video using IoU and area to pick the dominant slot.
        grp_slot_area = F.one_hot(grp_labels, num_classes=S).sum(dim=(1, 2, 3)).float()  # B x S
        iou_threshold = self.merge_iou_threshold
        eye_mask = torch.eye(S, device=features_device, dtype=torch.bool)
        # clone labels so that later steps use merged ids
        original_grp_labels = grp_labels.clone() # B T H W
        original_dec_labels = dec_labels.clone() # B T H W
        merged_grp_labels = grp_labels.clone()
        merged_dec_labels = dec_labels.clone()
        merged_dec_labels_for_eval = dec_labels_for_eval.clone() if dec_labels_for_eval is not None else None
        for b in range(B):
            valid_pairs = (slot_iou[b] > iou_threshold) & (~eye_mask)
            if not valid_pairs.any():
                continue
            areas_b = grp_slot_area[b]
            # Use all pairs above threshold to form the redundancy graph.
            redundant_pairs = valid_pairs.clone()
            redundant_pairs = redundant_pairs | redundant_pairs.t()
            # Find connected redundant groups and merge the whole cluster to the dominant slot.
            visited = torch.zeros(S, device=features_device, dtype=torch.bool)
            for i in range(S):
                if visited[i] or not redundant_pairs[i].any():
                    continue
                stack = [i]
                component = []
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    component.append(node)
                    neighbors = torch.nonzero(redundant_pairs[node]).flatten().tolist()
                    for n in neighbors:
                        if not visited[n]:
                            stack.append(int(n))
                if len(component) <= 1:
                    continue
                comp_tensor = torch.tensor(component, device=features_device)
                winner_idx = comp_tensor[areas_b[comp_tensor].argmax()].item()
                for loser_idx in component:
                    if loser_idx == winner_idx:
                        continue
                    merged_grp_labels[b][merged_grp_labels[b] == loser_idx] = winner_idx
                    merged_dec_labels[b][merged_dec_labels[b] == loser_idx] = winner_idx
                    if merged_dec_labels_for_eval is not None:
                        merged_dec_labels_for_eval[b][merged_dec_labels_for_eval[b] == loser_idx] = winner_idx
        grp_labels = merged_grp_labels
        dec_labels = merged_dec_labels
        if merged_dec_labels_for_eval is not None:
            dec_labels_for_eval = merged_dec_labels_for_eval
        

        # Pad the H and W dimensions to easily handle borders
        dec_padded_labels = torch.nn.functional.pad(dec_labels.float(), (1, 1, 1, 1, 1, 1), mode='replicate').long()
        grp_padded_labels = torch.nn.functional.pad(grp_labels.float(), (1, 1, 1, 1, 1, 1), mode='replicate').long()
    
        dec_neighbors = self.get_neighbor_patches(dec_padded_labels)
        grp_neighbors = self.get_neighbor_patches(grp_padded_labels)

        
        diff_neighbors_count = torch.zeros_like(dec_labels) # B T H W
        grp_diff_neighbors_count = torch.zeros_like(grp_labels)
        for grp_neighbor, dec_neighbor in zip(grp_neighbors, dec_neighbors):
            diff_neighbors_count += (dec_labels != dec_neighbor)
            grp_diff_neighbors_count += (grp_labels != grp_neighbor)
            

        
        # is_boundary = is_boundary & is_not_noise
        # Find the indices where the number of different neighbors is >= n
        is_boundary = grp_diff_neighbors_count >= dif_n_bd
        is_not_noise = (len(grp_neighbors) - grp_diff_neighbors_count) >= dif_n_bd_noise
        is_boundary = is_boundary & is_not_noise
        is_not_boundary = diff_neighbors_count < dif_n_nbd

        ### Labels from GRP (boundary) and DEC (non-boundary)
        grp_boundary_labels_list = []
        dec_not_boundary_labels_list = []
        ### Decoder prediction
        dec_boundary_pred_list = []
        dec_not_boundary_pred_list = []
        ### Grouper prediction
        grp_boundary_pred_list = []
        grp_not_boundary_pred_list = []

        for b in range(B):
            for t in range(T):
                ### label.
                # For boundaries, use GRP to make label
                grp_boundaries_labels = torch.masked_select(grp_labels[b, t], is_boundary[b, t])
                grp_boundary_labels_list.append(grp_boundaries_labels)
                # For non-boundaries, use DEC to make label
                dec_not_boundaries_labels = torch.masked_select(dec_labels[b, t], is_not_boundary[b, t])
                dec_not_boundary_labels_list.append(dec_not_boundaries_labels)

                ### Decoder Predictions.
                # Decoder Predictions for boundaries
                dec_boundaries_preds = dec_mask_hw[b, t][is_boundary[b, t]]
                dec_boundary_pred_list.append(dec_boundaries_preds)
                # Decoder Predictions for non-boundaries
                dec_not_boundaries_preds = dec_mask_hw[b, t][is_not_boundary[b, t]]
                dec_not_boundary_pred_list.append(dec_not_boundaries_preds)

                ### Grouping Predictions.
                # Grouper Predictions for boundaries
                grp_boundaries_preds = grp_mask_hw[b, t][is_boundary[b, t]]
                grp_boundary_pred_list.append(grp_boundaries_preds)
                # Grouper Predictions for non-boundaries
                grp_not_boundaries_preds = grp_mask_hw[b, t][is_not_boundary[b, t]]
                grp_not_boundary_pred_list.append(grp_not_boundaries_preds)


        ### labels for Decoder: Boundary w/ GRP, Non-Boundary w/ DEC
        grp_bd_labels = torch.cat(grp_boundary_labels_list, dim=0)
        dec_nbd_labels = torch.cat(dec_not_boundary_labels_list, dim=0)

        dec_bd_preds = torch.cat(dec_boundary_pred_list, dim=0)
        grp_not_bd_preds = torch.cat(grp_not_boundary_pred_list, dim=0)

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
            "grp_bd_labels": grp_bd_labels,
            "dec_bd_preds": dec_bd_preds,
            "dec_nbd_labels": dec_nbd_labels,
            "grp_nbd_preds": grp_not_bd_preds,
            "is_boundary": is_boundary,
            "is_not_boundary": is_not_boundary,
            "merged_grp_labels": grp_labels,
            "merged_dec_labels": dec_labels,
            "original_is_boundary": ori_is_boundary,
            "original_is_not_boundary": ori_is_not_boundary,
            "original_grp_labels": original_grp_labels,
            "original_dec_labels": original_dec_labels,
        }
        outputs["targets"] = self.get_targets(inputs, outputs)

        return outputs

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if masks is None:
            return None, None, None

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                masks_for_metrics = resizer(masks, target_masks)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    def get_neighbor_patches(self, padded_labels):
        # Spatial Neighbors (8-connectivity)
        top_left = padded_labels[:, 1:-1, :-2, :-2]
        top_center = padded_labels[:, 1:-1, :-2, 1:-1]
        top_right = padded_labels[:, 1:-1, :-2, 2:]
        middle_left = padded_labels[:, 1:-1, 1:-1, :-2]
        middle_right = padded_labels[:, 1:-1, 1:-1, 2:]
        bottom_left = padded_labels[:, 1:-1, 2:, :-2]
        bottom_center = padded_labels[:, 1:-1, 2:, 1:-1]
        bottom_right = padded_labels[:, 1:-1, 2:, 2:]

        # Temporal Neighbors
        prev_time = padded_labels[:, :-2, 1:-1, 1:-1]
        next_time = padded_labels[:, 2:, 1:-1, 1:-1]

        neighbors = [
            top_left, top_center, top_right,
            middle_left, middle_right,
            bottom_left, bottom_center, bottom_right,
            prev_time, next_time
        ]
        return neighbors # shape : list of tensors with shape B T H W
    
    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )

        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        for name, loss_fn in self.loss_fns.items():
            if name == "loss_grp2deccls" or name == "loss_dec2grpcls":
                if self.trainer.global_step < self.synergistic_warmup_ratio * self.max_steps:
                    continue
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]
            losses[name] = loss_fn(prediction, target)

        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        if 'coco' in self.experiment_name or 'voc' in self.experiment_name:
            batch['image'] = batch['image'].unsqueeze(1)
            batch['video'] = batch['image']
            if "segmentations" in batch:
                batch['segmentations'] = batch['segmentations'].unsqueeze(1)
        
        outputs = self.forward(batch)
        
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"train/{name}": loss for name, loss in losses.items()}
            to_log["train/loss"] = total_loss

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                values = metric(**batch, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        del outputs  # Explicitly delete to save memory

        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_inputs(
                batch[self.input_key],
                {key: aux_outputs[f"{key}_hard"] for key in self.mask_keys_to_visualize},
                mode="train",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if 'coco' in self.experiment_name or 'voc' in self.experiment_name:
            batch['image'] = batch['image'].unsqueeze(1)
            batch['video'] = batch['image']
            if "segmentations" in batch:
                batch['segmentations'] = batch['segmentations'].unsqueeze(1)
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch, train=False)
        aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        if self.val_metrics:
            for metric in self.val_metrics.values():
                metric.update(**batch, **outputs, **aux_outputs)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )
        gs = self.trainer.global_step

        ### Visualization Code for Attention map & Decoder Object map
        masks_to_vis = {"decoder_masks": aux_outputs[f"decoder_masks_vis_hard"].detach().clone()}
        self._save_inputs(batch[self.input_key], 
                        masks_to_vis, 
                        batch_idx, 
                        self.vis_cnt, 
                        merged_labels=outputs["merged_dec_labels"],
                        ori_labels=outputs["original_dec_labels"],
                        is_not_boundary=outputs["is_not_boundary"], 
                        ori_is_not_boundary=outputs["original_is_not_boundary"],
                        mode="val")
        
        masks_to_vis = {"grouping_masks": aux_outputs[f"grouping_masks_vis_hard"].detach().clone()}
        self._save_grouping_inputs(batch[self.input_key], 
                                masks_to_vis, 
                                batch_idx, self.vis_cnt, 
                                merged_labels=outputs["merged_grp_labels"],
                                ori_labels=outputs["original_grp_labels"],
                                is_boundary=outputs["is_boundary"], 
                                ori_is_boundary=outputs["original_is_boundary"],
                                mode="val")
        self.vis_cnt += 1
        ### Visualization End ###
        


        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            masks_to_vis = {
                key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
            }
            if batch["segmentations"].shape[-2:] != batch[self.input_key].shape[-2:]:
                masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                    batch["segmentations"], batch[self.input_key]
                )
            else:
                masks_to_vis["segmentations"] = batch["segmentations"]
            self._log_inputs(
                batch[self.input_key],
                masks_to_vis,
                mode="val",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")

    def visualize_video_frames_row(self, video, ori_masks, ori_video_with_masks, ori_video_with_masks_bd, merged_masks, merged_video_with_masks, merged_video_with_masks_bd, save_path, batch_idx=0):
        video_batch = video[batch_idx]  # (frame, channel, h, w)
        ori_mask_batch = ori_masks[batch_idx]
        ori_video_with_masks_batch = ori_video_with_masks[batch_idx]  # (frame, channel, h, w)
        ori_video_with_masks_bd_batch = ori_video_with_masks_bd[batch_idx]
        merged_mask_batch = merged_masks[batch_idx]
        merged_video_with_masks_batch = merged_video_with_masks[batch_idx]
        merged_video_with_masks_bd_batch = merged_video_with_masks_bd[batch_idx]

        num_frames = video_batch.shape[0]
        fig, axes = plt.subplots(7, num_frames, figsize=(num_frames * 3, 10))

        if num_frames == 1:
            axes = axes.reshape(7, 1)
        for frame_idx in range(num_frames):
            video_frame = video_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            video_frame = torch.clamp(video_frame, 0, 1)  #
            axes[0, frame_idx].imshow(video_frame.cpu().numpy())
            axes[0, frame_idx].set_title(f'Original {frame_idx}')
            axes[0, frame_idx].axis('off')

            ori_masks = ori_mask_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            ori_masks = torch.clamp(ori_masks, 0, 1)
            axes[1, frame_idx].imshow(ori_masks.cpu().numpy())
            axes[1, frame_idx].set_title(f'Original Masks {frame_idx}')
            axes[1, frame_idx].axis('off')

            ori_video_masks = ori_video_with_masks_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            ori_video_masks = torch.clamp(ori_video_masks, 0, 1)
            axes[2, frame_idx].imshow(ori_video_masks.cpu().numpy())
            axes[2, frame_idx].set_title(f'Original With Masks {frame_idx}')
            axes[2, frame_idx].axis('off')
            
            ori_video_masks_bd = ori_video_with_masks_bd_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            ori_video_masks_bd = torch.clamp(ori_video_masks_bd, 0, 1)
            axes[3, frame_idx].imshow(ori_video_masks_bd.cpu().numpy())
            axes[3, frame_idx].set_title(f'Original With Masks BD {frame_idx}')
            axes[3, frame_idx].axis('off')
            
            merged_masks = merged_mask_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            merged_masks = torch.clamp(merged_masks, 0, 1)
            axes[4, frame_idx].imshow(merged_masks.cpu().numpy())
            axes[4, frame_idx].set_title(f'Merged Masks {frame_idx}')
            axes[4, frame_idx].axis('off')
            
            merged_video_masks = merged_video_with_masks_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            merged_video_masks = torch.clamp(merged_video_masks, 0, 1)
            axes[5, frame_idx].imshow(merged_video_masks.cpu().numpy())
            axes[5, frame_idx].set_title(f'Merged With Masks {frame_idx}')
            axes[5, frame_idx].axis('off')
            
            merged_video_masks_bd = merged_video_with_masks_bd_batch[frame_idx].permute(1, 2, 0)  # (h, w, c)
            merged_video_masks_bd = torch.clamp(merged_video_masks_bd, 0, 1)
            axes[6, frame_idx].imshow(merged_video_masks_bd.cpu().numpy())
            axes[6, frame_idx].set_title(f'Merged With Masks BD {frame_idx}')
            axes[6, frame_idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {save_path}")

    def _save_inputs(
            self,
            inputs: torch.Tensor,
            masks_by_name: Dict[str, torch.Tensor],
            batch_idx: int,
            vis_cnt: int,
            mode: str,
            merged_labels=None,
            ori_labels=None,
            is_not_boundary=None,
            ori_is_not_boundary=None,
            step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            
            
            H, W = video.shape[-2], video.shape[-1]
            ori_labels = (F.interpolate(ori_labels.view(-1, 1, ori_labels.shape[-2], ori_labels.shape[-1]).float(),
                size=(H, W),
                mode="nearest",
            ).view(ori_labels.shape[0], ori_labels.shape[1], H, W).long())
            ori_labels = F.one_hot(ori_labels, num_classes=self.slot_num).permute(0,1,4,2,3).float()
            
            merged_labels = (F.interpolate(merged_labels.view(-1, 1, merged_labels.shape[-2], merged_labels.shape[-1]).float(),
                size=(H, W),
                mode="nearest",
            ).view(merged_labels.shape[0], merged_labels.shape[1], H, W).long())
            merged_labels = F.one_hot(merged_labels, num_classes=self.slot_num).permute(0,1,4,2,3).float()
        
            
            
            ori_video_with_masks = visualizations.mix_videos_with_masks(video, ori_labels.detach().clone()) # 1 30 3 518 518 # B T C H W
            merged_video_with_masks = visualizations.mix_videos_with_masks(video, merged_labels.detach().clone()) # 1 30 3 518 518 # B T C H W
            mask_name = "decoder"
            ori_video_with_masks_bd = self.apply_boundary(ori_video_with_masks.detach().clone(), ori_is_not_boundary)
            merged_video_with_masks_bd = self.apply_boundary(merged_video_with_masks.detach().clone(), is_not_boundary)


            ori_only_masks = visualizations.mix_videos_with_masks(torch.zeros_like(video).to(video.device), ori_labels.detach().clone(), alpha=1.0)
            merged_only_masks = visualizations.mix_videos_with_masks(torch.zeros_like(video).to(video.device), merged_labels.detach().clone(), alpha=1.0)
            ori_video_with_masks = ori_video_with_masks / 255.0
            ori_video_with_masks_bd = ori_video_with_masks_bd / 255.0
            ori_only_masks = ori_only_masks / 255.0
            merged_video_with_masks = merged_video_with_masks / 255.0
            merged_video_with_masks_bd = merged_video_with_masks_bd / 255.0
            merged_only_masks = merged_only_masks / 255.0
            
            dir = "../visualization/decoder/" + self.experiment_name + "/" + self.experiment_group + "/" + str(self.trainer.global_step)
            if not os.path.exists(dir):
                os.makedirs(dir)
            save_path = f"{dir}/{mode}_{self.input_key}_{mask_name}_frames_{vis_cnt}.png"
            self.visualize_video_frames_row(video, ori_only_masks, ori_video_with_masks, ori_video_with_masks_bd, merged_only_masks, merged_video_with_masks, merged_video_with_masks_bd, save_path)

    def _save_grouping_inputs(
            self,
            inputs: torch.Tensor,
            masks_by_name: Dict[str, torch.Tensor],
            batch_idx: int,
            vis_cnt: int,
            mode: str,
            merged_labels=None,
            ori_labels=None,
            is_boundary=None,
            ori_is_boundary=None,
            step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            
            H, W = video.shape[-2], video.shape[-1]
            ori_labels = (F.interpolate(ori_labels.view(-1, 1, ori_labels.shape[-2], ori_labels.shape[-1]).float(),
                size=(H, W),
                mode="nearest",
            ).view(ori_labels.shape[0], ori_labels.shape[1], H, W).long())
            ori_labels = F.one_hot(ori_labels, num_classes=self.slot_num).permute(0,1,4,2,3).float()
            
            merged_labels = (F.interpolate(merged_labels.view(-1, 1, merged_labels.shape[-2], merged_labels.shape[-1]).float(),
                size=(H, W),
                mode="nearest",
            ).view(merged_labels.shape[0], merged_labels.shape[1], H, W).long())
            merged_labels = F.one_hot(merged_labels, num_classes=self.slot_num).permute(0,1,4,2,3).float()
        
            
            
            ori_video_with_masks = visualizations.mix_videos_with_masks(video, ori_labels.detach().clone()) # 1 30 3 518 518 # B T C H W
            merged_video_with_masks = visualizations.mix_videos_with_masks(video, merged_labels.detach().clone()) # 1 30 3 518 518 # B T C H W
            mask_name = "grouping"
            ori_video_with_masks_bd = self.apply_boundary(ori_video_with_masks.detach().clone(), ori_is_boundary)
            merged_video_with_masks_bd = self.apply_boundary(merged_video_with_masks.detach().clone(), is_boundary)

            ori_only_masks = visualizations.mix_videos_with_masks(torch.zeros_like(video).to(video.device), ori_labels.detach().clone(), alpha=1.0)
            merged_only_masks = visualizations.mix_videos_with_masks(torch.zeros_like(video).to(video.device), merged_labels.detach().clone(), alpha=1.0)
            ori_video_with_masks = ori_video_with_masks / 255.0
            ori_video_with_masks_bd = ori_video_with_masks_bd / 255.0
            ori_only_masks = ori_only_masks / 255.0
            merged_video_with_masks = merged_video_with_masks / 255.0
            merged_video_with_masks_bd = merged_video_with_masks_bd / 255.0
            merged_only_masks = merged_only_masks / 255.0

            dir = "../visualization/grouping/" + self.experiment_name + "/" + self.experiment_group + "/" + str(self.trainer.global_step)
            if not os.path.exists(dir):
                os.makedirs(dir)
            save_path = f"{dir}/{mode}_{self.input_key}_{mask_name}_frames_{vis_cnt}.png"
            self.visualize_video_frames_row(video, ori_only_masks, ori_video_with_masks, ori_video_with_masks_bd, merged_only_masks, merged_video_with_masks, merged_video_with_masks_bd, save_path)
                
    def apply_boundary(self, ori_video_with_masks, is_not_boundary):
        B, T, C, H, W = ori_video_with_masks.shape
        Grid_H, Grid_W = is_not_boundary.shape[2], is_not_boundary.shape[3]
        
        red_val = 255.0
        red = torch.tensor([red_val, 0.0, 0.0], device=ori_video_with_masks.device).view(1, 1, 3, 1, 1)
        
        thickness = 2

        for b in range(B):
            for t in range(T):
                grid_mask = is_not_boundary[b, t] 
                true_indices = grid_mask.nonzero(as_tuple=False)
                
                for coords in true_indices:
                    gy, gx = coords[0].item(), coords[1].item()
                    
                    y_start = (gy * H) // Grid_H
                    y_end = ((gy + 1) * H + Grid_H - 1) // Grid_H
                    x_start = (gx * W) // Grid_W
                    x_end = ((gx + 1) * W + Grid_W - 1) // Grid_W

                    y_start = max(0, y_start)
                    y_end = min(H, y_end)
                    x_start = max(0, x_start)
                    x_end = min(W, x_end)
                    if y_end <= y_start or x_end <= x_start:
                        continue

                    local_thickness = min(thickness, y_end - y_start, x_end - x_start)

                    ori_video_with_masks[b, t, :, y_start:y_start+local_thickness, x_start:x_end] = red
                    ori_video_with_masks[b, t, :, y_end-local_thickness:y_end, x_start:x_end] = red
                    ori_video_with_masks[b, t, :, y_start:y_end, x_start:x_start+local_thickness] = red
                    ori_video_with_masks[b, t, :, y_start:y_end, x_end-local_thickness:x_end] = red
        return ori_video_with_masks
        

    def validation_epoch_end(self, outputs):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values

    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks",),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            if "video" in types:
                logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                data = data[:, :num_frames]
                data = data.flatten(0, 1)
                logger.experiment.add_image(
                    f"{name}/frames", make_grid(data, nrow=num_frames), global_step=global_step
                )

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples), global_step=global_step
            )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]

        return output

    def _get_tensorboard_logger(self):
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
