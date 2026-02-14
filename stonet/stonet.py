from __future__ import annotations
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from stonet.utils.pairing import attach_gt_pairs_inplace


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(node, "get"):
            return node.get(key, default)
        return getattr(node, key)
    except Exception:
        return default


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


@META_ARCH_REGISTRY.register()
class StoNet(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_pairs: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float, ...],
        pixel_std: Tuple[float, ...],
        semantic_on: bool,
        instance_on: bool,
        panoptic_on: bool,
        test_topk_per_image: int,
        pair_interleave: bool = True,
        main_class_ids: Optional[Sequence[int]] = None,
        sub_class_ids: Optional[Sequence[int]] = None,
        pair_contain_thr: float = 0.7,
        pair_iou_thr: float = 0.0,
        pair_alpha_iou: float = 0.15,
        pair_beta_center: float = 0.25,
        pair_ambiguous_margin: float = 0.05,
        pair_use_global_assignment: bool = True,
        force_role_gating: bool = True,
        gt_pairing_enable: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_pairs
        self.num_pairs = num_pairs
        self.num_query_tokens = 2 * self.num_pairs

        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata

        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self._backbone_checked = False
        self.pair_interleave = pair_interleave
        self.main_class_ids = main_class_ids if main_class_ids is not None else []
        self.sub_class_ids = sub_class_ids if sub_class_ids is not None else []
        self.pair_contain_thr = pair_contain_thr
        self.pair_iou_thr = pair_iou_thr
        self.pair_alpha_iou = pair_alpha_iou
        self.pair_beta_center = pair_beta_center
        self.pair_ambiguous_margin = pair_ambiguous_margin
        self.pair_use_global_assignment = pair_use_global_assignment
        self.force_role_gating = force_role_gating
        self.gt_pairing_enable = gt_pairing_enable

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        stonet_cfg = cfg.MODEL.STONET
        decoder_name = stonet_cfg.TRANSFORMER_DECODER_NAME
        use_fastinst_baseline = str(decoder_name) == "FastInstDecoder"

        deep_supervision = stonet_cfg.DEEP_SUPERVISION
        no_object_weight = stonet_cfg.NO_OBJECT_WEIGHT
        class_weight = stonet_cfg.CLASS_WEIGHT
        dice_weight = stonet_cfg.DICE_WEIGHT
        mask_weight = stonet_cfg.MASK_WEIGHT
        location_weight = stonet_cfg.LOCATION_WEIGHT
        proposal_weight = stonet_cfg.PROPOSAL_WEIGHT


        # Matcher
        main_class_ids = _cfg_get(stonet_cfg, "MAIN_CLASS_IDS", [])
        sub_class_ids  = _cfg_get(stonet_cfg, "SUB_CLASS_IDS", [])
        cfg_role_split = _cfg_get(stonet_cfg, "MATCHER_ROLE_SPLIT", False)
        if use_fastinst_baseline:
            main_class_ids = []
            sub_class_ids = []
            cfg_role_split = False

        pair_interleave = _cfg_get(stonet_cfg, "PROPOSAL_INTERLEAVE", True)
        enable_role_split = cfg_role_split and (len(main_class_ids) > 0) and (len(sub_class_ids) > 0)
        allow_unk = _cfg_get(stonet_cfg, "MATCHER_ALLOW_UNKNOWN_GT_ROLE", True)

        gt_pairing_enable = _cfg_get(stonet_cfg, "GT_PAIRING_ENABLE", True)
        if use_fastinst_baseline:
            gt_pairing_enable = False

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_location=location_weight,
            num_points=stonet_cfg.TRAIN_NUM_POINTS,
            enable_role_split=enable_role_split,
            pair_interleave=pair_interleave,
            main_class_ids=main_class_ids,
            sub_class_ids=sub_class_ids,
            allow_unknown_gt_role=allow_unk,
        )

        # Criterion
        use_structure = stonet_cfg.USE_STRUCTURE_LOSS
        if use_structure and not gt_pairing_enable:
            raise ValueError("USE_STRUCTURE_LOSS = True requires GT_PAIRING_ENABLE = True (Otherwise structure loss will be silently 0).")
        enable_contain = stonet_cfg.STRUCTURE_ENABLE_CONTAIN
        enable_center = stonet_cfg.STRUCTURE_ENABLE_CENTER
        enable_area = stonet_cfg.STRUCTURE_ENABLE_AREA
        w_contain = stonet_cfg.STRUCTURE_WEIGHT_CONTAIN
        w_center = stonet_cfg.STRUCTURE_WEIGHT_CENTER
        w_area = stonet_cfg.STRUCTURE_WEIGHT_AREA
        structure_area_ratio_target = stonet_cfg.STRUCTURE_AREA_RATIO_TARGET
        structure_eps = stonet_cfg.STRUCTURE_EPS
        structure_apply_to_aux = stonet_cfg.STRUCTURE_APPLY_TO_AUX
        pair_interleave = stonet_cfg.PROPOSAL_INTERLEAVE
        structure_role_filter = stonet_cfg.STRUCTURE_ROLE_FILTER
        weight_dict: Dict[str, float] = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        if deep_supervision:
            dec_layers = stonet_cfg.DEC_LAYERS
            aux_weight_dict: Dict[str, float] = {}
            for i in range(2 * dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict.update({"loss_proposal": proposal_weight})
        losses: List[str] = ["labels", "masks"]
        if use_structure and (enable_contain or enable_center or enable_area):
            losses.append("structure")
            if enable_contain:
                weight_dict["loss_contain"] = w_contain
            if enable_center:
                weight_dict["loss_center"] = w_center
            if enable_area:
                weight_dict["loss_area"] = w_area
            if deep_supervision and structure_apply_to_aux:
                dec_layers = stonet_cfg.DEC_LAYERS
                for i in range(2 * dec_layers):
                    if enable_contain:
                        weight_dict[f"loss_contain_{i}"] = w_contain
                    if enable_center:
                        weight_dict[f"loss_center_{i}"] = w_center
                    if enable_area:
                        weight_dict[f"loss_area_{i}"] = w_area
        criterion_kwargs = dict(
            num_classes=sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=stonet_cfg.TRAIN_NUM_POINTS,
            oversample_ratio=stonet_cfg.OVERSAMPLE_RATIO,
            importance_sample_ratio=stonet_cfg.IMPORTANCE_SAMPLE_RATIO,
            area_ratio_target=structure_area_ratio_target,
            structure_eps=structure_eps,
            apply_structure_to_aux=structure_apply_to_aux,
            enable_structure_loss=use_structure,
            enable_loss_contain=enable_contain,
            enable_loss_center=enable_center,
            enable_loss_area=enable_area,
            pair_interleave=pair_interleave,
            enable_structure_role_filter=structure_role_filter,
        )
        criterion_kwargs = _filter_kwargs_for_callable(SetCriterion.__init__, criterion_kwargs)
        criterion = SetCriterion(**criterion_kwargs)

        if hasattr(sem_seg_head, "predictor") and hasattr(sem_seg_head.predictor, "criterion"):
            sem_seg_head.predictor.criterion = criterion
            assert sem_seg_head.predictor.criterion is criterion, "Failed to inject criterion into decoder predictor."
        else:
            raise AttributeError("sem_seg_head.predictor.criterion not found; IA-Guide requires decoder.criterion to be set.")

        pair_contain_thr = stonet_cfg.PAIR_CONTAIN_THR
        pair_iou_thr = stonet_cfg.PAIR_IOU_THR
        pair_alpha_iou = stonet_cfg.PAIR_ALPHA_IOU
        pair_beta_center = stonet_cfg.PAIR_BETA_CENTER
        pair_ambiguous_margin = stonet_cfg.PAIR_AMBIGUOUS_MARGIN
        pair_use_global_assignment = stonet_cfg.PAIR_USE_GLOBAL_ASSIGNMENT

        force_role_gating = stonet_cfg.FORCE_ROLE_GATING
        if use_fastinst_baseline:
            force_role_gating = False

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_pairs": stonet_cfg.NUM_OBJECT_QUERIES,
            "object_mask_threshold": stonet_cfg.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": stonet_cfg.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": stonet_cfg.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": bool(
                stonet_cfg.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or stonet_cfg.TEST.PANOPTIC_ON
                or stonet_cfg.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": stonet_cfg.TEST.SEMANTIC_ON,
            "instance_on": stonet_cfg.TEST.INSTANCE_ON,
            "panoptic_on": stonet_cfg.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "gt_pairing_enable": gt_pairing_enable,
            "pair_interleave": pair_interleave,
            "main_class_ids": main_class_ids,
            "sub_class_ids": sub_class_ids,
            "pair_contain_thr": pair_contain_thr,
            "pair_iou_thr": pair_iou_thr,
            "pair_alpha_iou": pair_alpha_iou,
            "pair_beta_center": pair_beta_center,
            "pair_ambiguous_margin": pair_ambiguous_margin,
            "pair_use_global_assignment": pair_use_global_assignment,
            "force_role_gating": force_role_gating,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        if not self._backbone_checked:
            self._backbone_checked = True
        if self.training:
            assert "instances" in batched_inputs[0], "Training requires 'instances' in inputs."
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            outputs = self.sem_seg_head(features, targets)
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        
        outputs = self.sem_seg_head(features)
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets: List[Dict[str, Any]] = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            if hasattr(gt_masks, "tensor"):
                gt_masks_t = gt_masks.tensor
            else:
                gt_masks_t = gt_masks
            padded_masks = torch.zeros(
                (gt_masks_t.shape[0], h_pad, w_pad),
                dtype=gt_masks_t.dtype,
                device=gt_masks_t.device,
            )
            padded_masks[:, : gt_masks_t.shape[1], : gt_masks_t.shape[2]] = gt_masks_t
            t: Dict[str, Any] = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            }
            if self.gt_pairing_enable:
                attach_gt_pairs_inplace(
                    t,
                    stomata_ids=self.main_class_ids,
                    pore_ids=self.sub_class_ids,
                    contain_thr=self.pair_contain_thr,
                    iou_thr=self.pair_iou_thr,
                    alpha_iou=self.pair_alpha_iou,
                    beta_center=self.pair_beta_center,
                    ambiguous_margin=self.pair_ambiguous_margin,
                    use_global_assignment=self.pair_use_global_assignment,
                )
            new_targets.append(t)
        return new_targets

    def _apply_role_class_gating(self, mask_cls_logits: torch.Tensor) -> torch.Tensor:
        if not self.force_role_gating:
            return mask_cls_logits
        if (not self.main_class_ids) or (not self.sub_class_ids):
            return mask_cls_logits
        logits = mask_cls_logits.clone()
        C = self.sem_seg_head.num_classes  # no-object index = C
        Q = logits.shape[0]
        valid_main = torch.zeros((C,), dtype=torch.bool, device=logits.device)
        valid_sub = torch.zeros((C,), dtype=torch.bool, device=logits.device)
        valid_main[torch.as_tensor(self.main_class_ids, device=logits.device)] = True
        valid_sub[torch.as_tensor(self.sub_class_ids, device=logits.device)] = True
        valid_main_full = torch.cat(
            [valid_main, torch.ones((1,), dtype=torch.bool, device=logits.device)], dim=0
        )
        valid_sub_full = torch.cat(
            [valid_sub, torch.ones((1,), dtype=torch.bool, device=logits.device)], dim=0
        )
        if self.pair_interleave:
            main_rows = torch.arange(0, Q, 2, device=logits.device)
            sub_rows = torch.arange(1, Q, 2, device=logits.device)
        else:
            K = Q // 2
            main_rows = torch.arange(0, K, device=logits.device)
            sub_rows = torch.arange(K, Q, device=logits.device)
        allowed = torch.ones((Q, C + 1), dtype=torch.bool, device=logits.device)
        allowed[main_rows] = valid_main_full.unsqueeze(0)
        allowed[sub_rows] = valid_sub_full.unsqueeze(0)
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~allowed, neg_inf)
        return logits

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = self._apply_role_class_gating(mask_cls)
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        mask_cls = self._apply_role_class_gating(mask_cls)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info: List[Dict[str, Any]] = []
        current_segment_id = 0
        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list: Dict[int, int] = {}
        for k in range(cur_classes.shape[0]):
            pred_class = int(cur_classes[k].item())
            isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()

            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            if mask_area <= 0 or original_area <= 0 or mask.sum().item() <= 0:
                continue
            if mask_area / original_area < self.overlap_threshold:
                continue
            if not isthing:
                if pred_class in stuff_memory_list:
                    panoptic_seg[mask] = stuff_memory_list[pred_class]
                    continue
                stuff_memory_list[pred_class] = current_segment_id + 1
            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {"id": current_segment_id, "isthing": bool(isthing), "category_id": pred_class}
            )
        return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        image_size = mask_pred.shape[-2:]
        mask_cls = self._apply_role_class_gating(mask_cls)
        Q = mask_cls.shape[0]
        C = self.sem_seg_head.num_classes
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(C, device=self.device).unsqueeze(0).repeat(Q, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_query_indices = torch.div(topk_indices, C, rounding_mode="trunc")
        mask_pred = mask_pred[topk_query_indices]
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        mask_pred_sigmoid = mask_pred.sigmoid()
        result.pred_masks = (mask_pred_sigmoid > 0.5).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4, device=mask_pred.device))

        mask_scores_per_image = (mask_pred_sigmoid.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
            result.pred_masks.flatten(1).sum(1) + 1e-6
        )
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result