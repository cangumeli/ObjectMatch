from typing import Optional

import torch
import torch.nn as nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances

from nocpred.modeling.roi_heads.noc_head import build_noc_head


@ROI_HEADS_REGISTRY.register()
class NOCPredROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        noc_in_features: Optional[list[str]] = None,
        noc_pooler: Optional[ROIPooler] = None,
        noc_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noc_in_features = noc_in_features
        self.noc_pooler = noc_pooler
        self.noc_head = noc_head

    @property
    def without_nocs(self):
        return self.noc_head is None

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret: dict = super().from_config(cfg, input_shape)
        if not cfg.MODEL.ROI_HEADS.WITHOUT_NOCS:
            ret.update(cls._init_noc_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_noc_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES  # noqa: E22
        pooler_resolution = cfg.MODEL.ROI_NOC_HEAD.POOLER_RESOLUTION
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        in_channels = set(input_shape[f].channels for f in in_features)
        assert len(in_channels) == 1
        in_channels = in_channels.pop()
        input_shape = ShapeSpec(
            channels=in_channels,
            height=pooler_resolution,
            width=pooler_resolution,
        )

        return {
            'noc_in_features': in_features,
            'noc_pooler': ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            ),
            'noc_head': build_noc_head(cfg, input_shape),
        }

    def forward(
        self,
        images: ImageList,
        features: dict[str, torch.Tensor],
        proposals: list[Instances],
        targets: Optional[list[Instances]] = None,
    ) -> tuple[list[Instances], dict[str, torch.Tensor]]:
        instances, losses = super().forward(images, features, proposals, targets)
        if self.training:
            losses.update(self._forward_nocs(features, instances))
        else:
            instances = self._forward_nocs(features, instances)
        return instances, losses

    def _forward_nocs(self, features: dict[str, torch.Tensor], instances: list[Instances]):
        if self.without_nocs:
            return {} if self.training else instances

        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        features = [features[f] for f in self.noc_in_features]
        boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        features = self.noc_pooler(features, boxes)
        return self.noc_head(features, instances=instances)
