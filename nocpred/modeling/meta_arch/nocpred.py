import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import Backbone, build_backbone, GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class NOCPred(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        depth_backbone: Optional[Backbone] = None,
        normal_backbone: Optional[Backbone] = None,
        backbone_aggr: str = 'mean',
        remove_color: bool = False,
        backbone_drop: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth_backbone = depth_backbone
        self.normal_backbone = normal_backbone
        self.backbone_aggr = backbone_aggr
        self.backbone_drop = backbone_drop
        self.remove_color = remove_color
        self.should_replicate_for_depth = self.depth_backbone is not None
        self.should_replicate_for_normal = self.normal_backbone is not None

    @classmethod
    def from_config(cls, cfg) -> dict:
        dt: dict = super().from_config(cfg)
        assert cfg.INPUT.COLOR or cfg.INPUT.DEPTH or cfg.INPUT.NORMAL, 'No backbone inputs'
        if cfg.INPUT.DEPTH:
            dt['depth_backbone'] = build_backbone(cfg)
        if cfg.INPUT.NORMAL:
            dt['normal_backbone'] = build_backbone(cfg)
        if not cfg.INPUT.COLOR:
            dt['remove_color'] = True
        dt['backbone_aggr'] = cfg.MODEL.BACKBONE.FEATURE_AGGREGATION
        pdrop = cfg.MODEL.BACKBONE.FEATURE_DROP
        dt['backbone_drop'] = nn.Dropout2d(pdrop) if pdrop > 0 else nn.Identity()
        return dt

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], *args, **kwargs):
        names = ['depth', 'normal']
        for name in names:
            backbone_name = name + '_backbone'
            if getattr(self, backbone_name) is not None:
                setattr(
                    self,
                    'should_replicate_for_{}'.format(name),
                    not any(k.startswith(backbone_name) for k in state_dict),
                )
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self.replicate_backbone()
        return result

    def replicate_backbone(self):
        if self.should_replicate_for_depth:
            self.depth_backbone.load_state_dict(self.backbone.state_dict())
        if self.should_replicate_for_normal:
            self.normal_backbone.load_state_dict(self.backbone.state_dict())
        if self.remove_color:
            del self.backbone
            self.backbone = None

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        images, features = self.run_backbone(batched_inputs)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert 'proposals' in batched_inputs[0]
            proposals = [x['proposals'].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: list[dict[str, torch.Tensor]],
        detected_instances: Optional[list[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images, features = self.run_backbone(batched_inputs)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert 'proposals' in batched_inputs[0]
                proposals = [x['proposals'].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            raise RuntimeError('Pre-computed proposals are not supported yet')
            # detected_instances = [x.to(self.device) for x in detected_instances]
            # results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def run_backbone(
        self,
        batched_inputs: list[dict[str, torch.Tensor]],
    ) -> tuple[ImageList, dict[str, torch.Tensor]]:

        images = None
        feature_list: list[dict[str, torch.Tensor]] = []
        reference_sizes = None
        for backbone, field in zip(
            [self.backbone, self.depth_backbone, self.normal_backbone],
            ['image', 'depth_image', 'normal_image'],
        ):
            if backbone is not None:
                images_field = self.preprocess_image([{'image': x[field]} for x in batched_inputs])
                feats = backbone(images_field.tensor)

                sizes = {k: feats[k].shape[-2:] for k in feats}
                if reference_sizes is None or all(
                    all(s1 >= s2 for s1, s2 in zip(sizes[k], reference_sizes[k]))
                    for k in feats
                ):
                    reference_sizes = sizes

                feature_list.append(feats)
                if images is None:
                    images = images_field

        # Handle up-sampling
        for feats in feature_list:
            for k, (rh, rw) in reference_sizes.items():
                feat = feats[k]
                h, w = feat.shape[-2:]
                if (h, w) != (rh, rw):
                    scale = math.ceil(rh / h)
                    feat = F.interpolate(feat, scale_factor=scale)
                    h, w = feat.shape[-2:]
                    mod_h, mod_w = h - rh, w - rw
                    if mod_h != 0 or mod_w != 0:
                        start_y, start_x = (mod_h // 2), (mod_w // 2)
                        feat = feat[..., start_y:(start_y + rh), start_x:(start_x + rw)]
                    feats[k] = feat
                    # from IPython import embed; embed()

        if len(feature_list) == 1:
            features = feature_list[0]
            for k, v in features.items():
                features[k] = self.backbone_drop(v)
        else:
            # Collect features
            keys = list(feature_list[0].keys())
            joint_features: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
            for k in keys:
                for feat in feature_list:
                    joint_features[k].append(feat[k])

            # Aggregate features
            for k, v in joint_features.items():
                feat_stack = torch.reshape(self.backbone_drop(torch.cat(v)), (len(v), *v[0].shape))
                feat_aggr = getattr(torch, self.backbone_aggr)(feat_stack, dim=0)
                if not torch.is_tensor(feat_aggr):  # For max e.t.c.
                    feat_aggr = feat_aggr.values
                joint_features[k] = feat_aggr
            features = joint_features

        return images, features
