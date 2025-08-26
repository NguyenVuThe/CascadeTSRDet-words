# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import MyFastRCNNOutputLayers, fast_rcnn_inference
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
import copy
from transformers import BertTokenizer, BertModel
import logging

from text_embedding import TextEmbeddingLookup
import numpy as np
import json

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@ROI_HEADS_REGISTRY.register()
class MyCascadeROIHeads(StandardROIHeads):
    """
    The ROI heads that implement :paper:`Cascade R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_heads (list[nn.Module]): box head for each cascade stage
            box_predictors (list[nn.Module]): box predictor for each cascade stage
            proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
                match boxes with ground truth for each stage. The first matcher matches
                RPN proposals with ground truth, the other matchers use boxes predicted
                by the previous stage as proposals and match them with ground truth.
        """
        assert "proposal_matcher" not in kwargs, (
            "CascadeROIHeads takes 'proposal_matchers=' for each stage instead "
            "of one 'proposal_matcher='."
        )
        # The first matcher matches RPN proposals with ground truth, done in the base class
        kwargs["proposal_matcher"] = proposal_matchers[0]
        num_stages = self.num_cascade_stages = len(box_heads)
        box_heads = nn.ModuleList(box_heads)
        box_predictors = nn.ModuleList(box_predictors)
        assert len(box_predictors) == num_stages, f"{len(box_predictors)} != {num_stages}!"
        assert len(proposal_matchers) == num_stages, f"{len(proposal_matchers)} != {num_stages}!"
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_heads,
            box_predictor=box_predictors,
            **kwargs,
        )
        self.proposal_matchers = proposal_matchers
        self.num_classes = kwargs.get("num_classes", 7)

        # CUSTOM PART: BERT model for text embedding
        self.text_ffn = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fusion_ffn = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #self.proposal_features = nn.Embedding(512, 1024)
        self.text_lookup = TextEmbeddingLookup(
            emb_path="embeddings/embeddings.npy",
            vocab_path="embeddings/token2id.json"
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.pop("proposal_matcher")
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                MyFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def forward(self, images, features, proposals, targets=None, word_data_list=None):

        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        
        #proposal_list = []
        #for i, proposal in enumerate(proposals):
        #    if len(proposal) < 512:
        #        repetitions = 512 // len(proposal)
        #        proposal_new = copy.deepcopy(proposal)
        #        for _ in range(repetitions - 1):
        #            proposal_new = Instances.cat([proposal_new, proposal])
        #        pad_num = 512 - len(proposal_new)
        #        for i in range(pad_num):
        #            proposal_new = Instances.cat([proposal_new, proposal[i]])

        #        assert len(proposal_new) == 512
        #        proposal_list.append(proposal_new)
        #    else:
        #        #assert len(proposal) == 512
        #        proposal_list.append(proposal)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets, word_data_list=word_data_list)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, word_data_list=word_data_list)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None, word_data_list=None):
        """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        bs = len(proposals)

        #init_features = self.proposal_features.weight[None].repeat(1, bs, 1)
        #proposal_features = init_features.clone()

        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            #predictions, proposal_features = self._run_stage(features, proposal_features, proposals, k)
            predictions = self._run_stage(features, proposals, k, word_data_list)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
                self.num_classes,
            )
            return pred_instances

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    #def _run_stage(self, features, proposal_features, proposals, stage):
    # def _run_stage(self, features, proposals, stage, word_data_list=None):
    #     """
    #     Args:
    #         features (list[Tensor]): #lvl input features to ROIHeads
    #         proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
    #         stage (int): the current stage

    #     Returns:
    #         Same output as `FastRCNNOutputLayers.forward()`.
    #     """
    #     proposal_boxes = [x.proposal_boxes for x in proposals]
    #     logging.debug(f"[Stage {stage}] Running RoIAlign on {len(proposal_boxes)} images.")
    #     logging.debug(f"[Stage {stage}] Proposal boxes per image: {[len(p) for p in proposal_boxes]}")
    #     box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
    #     logging.debug(f"[Stage {stage}] RoIAligned feature shape: {box_features.shape}")
    #     # The original implementation averages the losses among heads,
    #     # but scale up the parameter gradients of the heads.
    #     # This is equivalent to adding the losses among heads,
    #     # but scale down the gradients on features.
    #     if self.training:
    #         box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
    #         logging.debug(f"[Stage {stage}] Passing box features through box_head.")

    #     #proposal feature shape: 1, 1024, 1024
    #     #box features shape: 1024, 256, 7, 7

    #     #box_features, proposal_features = self.box_head[stage](box_features, proposal_features)
    #     #return self.box_predictor[stage](box_features, proposal_features), proposal_features
    #     box_features  = self.box_head[stage](box_features)
    #     logging.debug(f"[Stage {stage}] Box head output shape: {box_features.shape}")

    #     if stage == 2 and word_data_list is not None:
    #         fused_box_features = []
    #         start_idx = 0
    #         for img_idx, (proposal, word_data) in enumerate(zip(proposals, word_data_list)):
    #             proposal_boxes = proposal.proposal_boxes.tensor  # [N, 4]
    #             logging.debug(f"[Stage {stage}] Processing image {img_idx} with {len(proposal_boxes)} proposals.")
    #             num_props = proposal_boxes.shape[0]
    #             logging.debug(f"[Stage {stage}] Number of proposals: {num_props}")
    #             visual_feats = box_features[start_idx : start_idx + num_props]  # [N, C]
    #             logging.debug(f"[Stage {stage}] Visual features shape: {visual_feats.shape}")
    #             start_idx += num_props

    #             # Text embedding for each proposal
    #             text_embeds = []
    #             for i, box in enumerate(proposal_boxes):
    #                 words = self.get_words_in_box(box.cpu().numpy(), word_data)
    #                 logging.debug(f"[Stage {stage}] Words in box {i}: {words}")
    #                 sentence = " ".join(words)
    #                 logging.debug(f"[Stage {stage}] Sentence for box {i}: {sentence}")
    #                 # If no words found, use a placeholder
    #                 if sentence.strip() == "":
    #                     sentence = "[PAD]"
    #                 # Tokenize and get BERT embeddings
    #                 inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(visual_feats.device)
    #                 logging.debug(f"[Stage {stage}] Tokenized inputs for box {i}: {inputs}")
    #                 # Get BERT embeddings
    #                 with torch.no_grad():
    #                     outputs = self.bert_model(**inputs)
    #                 logging.debug(f"[Stage {stage}] BERT outputs for box {i}: {outputs.last_hidden_state.shape}")
    #                 pooled = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
    #                 logging.debug(f"[Stage {stage}] Pooled text embedding for box {i}: {pooled.shape}")
    #                 text_embeds.append(pooled)
    #             text_feats = torch.cat(text_embeds, dim=0)  # [N, 768]
    #             logging.debug(f"[Stage {stage}] Concatenated text features shape: {text_feats.shape}")
    #             text_feats = self.text_ffn(text_feats)      # [N, 1024]
    #             logging.debug(f"[Stage {stage}] Text features after FFN shape: {text_feats.shape}")

    #             # Concatenate visual and text features
    #             fusion_input = torch.cat([visual_feats, text_feats], dim=1)  # [N, 2048]
    #             logging.debug(f"[Stage {stage}] Fusion input shape: {fusion_input.shape}")
    #             fused_feats = self.fusion_ffn(fusion_input)                  # [N, 1024]
    #             logging.debug(f"[Stage {stage}] Fused features shape: {fused_feats.shape}")
    #             fused_box_features.append(fused_feats)
    #         box_features = torch.cat(fused_box_features, dim=0)  # [total_props, 1024]
    #         logging.debug(f"[Stage {stage}] Final box features shape after fusion: {box_features.shape}")

    #         logging.debug(f"Output: {self.box_predictor[stage]}")
            
    #     return self.box_predictor[stage](box_features)

    def _run_stage(self, features, proposals, stage, word_data_list=None):
        proposal_boxes = [x.proposal_boxes for x in proposals]
        logging.debug(f"[Stage {stage}] Running RoIAlign on {len(proposal_boxes)} images.")
        logging.debug(f"[Stage {stage}] Proposal boxes per image: {[len(p) for p in proposal_boxes]}")
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        logging.debug(f"[Stage {stage}] RoIAligned feature shape: {box_features.shape}")

        if self.training:
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
            logging.debug(f"[Stage {stage}] Passing box features through box_head.")

        box_features  = self.box_head[stage](box_features)
        logging.debug(f"[Stage {stage}] Box head output shape: {box_features.shape}")

        if stage == 0 and word_data_list is not None:
            fused_box_features = []
            start_idx = 0
            for img_idx, (proposal, word_data) in enumerate(zip(proposals, word_data_list)):
                proposal_boxes = proposal.proposal_boxes.tensor  # [N, 4]
                logging.debug(f"[Stage {stage}] Processing image {img_idx} with {len(proposal_boxes)} proposals.")
                num_props = proposal_boxes.shape[0]
                logging.debug(f"[Stage {stage}] Number of proposals: {num_props}")
                visual_feats = box_features[start_idx : start_idx + num_props]  # [N, C]
                logging.debug(f"[Stage {stage}] Visual features shape: {visual_feats.shape}")
                start_idx += num_props

                # Text embedding for each proposal
                text_embeds = []
                for i, box in enumerate(proposal_boxes):
                    words = self.get_words_in_box(box.cpu().numpy(), word_data, drop_symbols=True)
                    if len(words) == 0:
                        pooled = torch.zeros(self.text_lookup.dim, device=visual_feats.device)
                    else:
                        logging.debug(f"[Stage {stage}] Words in box {i}: {words}")
                        embs = [self.text_lookup.get(w) for w in words]
                        pooled_np = np.mean(embs, axis=0)
                        pooled = torch.tensor(pooled_np, device=visual_feats.device, dtype=visual_feats.dtype)

                    text_embeds.append(pooled.unsqueeze(0)) # [1, 768]

                text_feats = torch.cat(text_embeds, dim=0)  # [N, 768]
                logging.debug(f"[Stage {stage}] Concatenated text features shape: {text_feats.shape}")
                text_feats = self.text_ffn(text_feats)      # [N, 1024]
                logging.debug(f"[Stage {stage}] Text features after FFN shape: {text_feats.shape}")  
                # Concatenate visual and text features
                fusion_input = torch.cat([visual_feats, text_feats], dim=1)  # [N, 2048]
                logging.debug(f"[Stage {stage}] Fusion input shape: {fusion_input.shape}")
                fused_feats = self.fusion_ffn(fusion_input)                  # [N, 1024]
                logging.debug(f"[Stage {stage}] Fused features shape: {fused_feats.shape}")
                fused_box_features.append(fused_feats)

            box_features = torch.cat(fused_box_features, dim=0)  # [total_props, 1024]
            logging.debug(f"[Stage {stage}] Final box features shape after fusion: {box_features.shape}")

        return self.box_predictor[stage](box_features)

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)

        #proposal_list = []
        #for i, proposal in enumerate(proposals):
        #    if len(proposal) < 512:
        #        repetitions = 512 // len(proposal)
        #        proposal_new = copy.deepcopy(proposal)
        #        for _ in range(repetitions - 1):
        #            proposal_new = Instances.cat([proposal_new, proposal])
        #        pad_num = 512 - len(proposal_new)
        #        for i in range(pad_num):
        #            proposal_new = Instances.cat([proposal_new, proposal[i]])
        #        proposal_list.append(proposal_new)
        #    else:
        #        proposal_list.append(proposal)

        return proposals
    
    def get_words_in_box(self, box, word_data, min_len: int = 1, drop_symbols: bool = True):
        x1, y1, x2, y2 = box
        words = []
        for word in word_data:
            wx1, wy1, wx2, wy2 = word["bbox"]
            token = word["text"].strip()

            # Check word nằm trong bbox
            if x1 <= wx1 and y1 <= wy1 and x2 >= wx2 and y2 >= wy2:
                # Bỏ token quá ngắn
                if len(token) < min_len:
                    continue
                
                # Nếu drop_symbols = True → bỏ token toàn ký tự .,;:-_
                if drop_symbols and all(ch in ". ,;:-_`'\"~" for ch in token):
                    continue

                words.append(token)
        
        return words
