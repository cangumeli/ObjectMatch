from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import detectron2.layers as L
from detectron2.config import configurable
from detectron2.structures import Boxes, Instances
from detectron2.utils.registry import Registry
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.transforms import euler_angles_to_matrix

from nocpred.structures import (
    DepthPoints,
    MeshGrids,
    NOCs,
    RotationMats,
    Rotations,
    Scales,
    transform_grid,
    Translations,
)


ROI_NOC_HEAD_REGISTRY = Registry('ROI_NOC_HEAD')


@ROI_NOC_HEAD_REGISTRY.register()
class ConvUpsampleNOCHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        num_pre_convs: int = 4,
        num_post_convs: int = 1,
        num_mlp_hiddens: int = 2,
        filter_size: int = 3,
        num_cond: int = 0,
        num_mlp_cond: int = 0,
        use_deconv: bool = False,
        use_pixel_shuffle: bool = True,
        output_size: int = 3,
        noc_loss_weight: float = 1.,
        proc_loss: bool = False,
        proc_rot_weight: float = 1.,
        proc_trans_weight: float = 1.,
        min_proc: int = 10,
        eval_mask_thresh: float = 0.5,
        predict_scale: bool = False,
        predict_alignment: bool = False,
        num_classes: int = 9,
        pooler_size: tuple[int, int] = (16, 16),
    ):
        super().__init__()
        padding = filter_size // 2 - (1 - filter_size % 2)

        self.pre_convs = nn.Sequential(*[
            L.Conv2d(
                input_size if i == 0 else hidden_size,
                hidden_size,
                filter_size,
                padding=padding,
                activation=nn.ReLU(True),
            )
            for i in range(num_pre_convs)
        ])

        if use_deconv:
            self.upsample = nn.Sequential(
                L.ConvTranspose2d(hidden_size, hidden_size, 2, stride=2),
                nn.ReLU(True),
            )
        elif use_pixel_shuffle:
            self.upsample = nn.Sequential(
                L.Conv2d(
                    hidden_size,
                    hidden_size * 4,
                    # 1,
                    filter_size,
                    padding=padding,
                    activation=nn.ReLU(True),
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.post_convs = nn.Sequential(*[
            L.Conv2d(
                hidden_size + (i == 0) * num_cond,
                hidden_size,
                filter_size,
                padding=padding,
                activation=nn.ReLU(True),
            )
            for i in range(num_post_convs)
        ])

        self.shared_mlp = nn.Sequential(*[
            L.Conv2d(
                hidden_size + (i == 0) * num_mlp_cond,
                hidden_size,
                1,
                activation=nn.ReLU(True),
            )
            for i in range(num_mlp_hiddens)
        ])

        self.output = L.Conv2d(hidden_size, output_size, 1)

        self.noc_loss_weight = noc_loss_weight
        self.loss = nn.L1Loss(reduction='none')

        self.proc_loss = proc_loss and (proc_rot_weight > 0 or proc_trans_weight > 0)
        self.proc_rot_weight = proc_rot_weight
        self.proc_trans_weight = proc_trans_weight
        self.min_proc = min_proc
        self.eval_mask_thresh = eval_mask_thresh

        if self.proc_loss:
            self.rot_loss = nn.L1Loss()
            self.trans_loss = nn.HuberLoss()

        self.predict_alignmnet = predict_alignment
        self.predict_scale = predict_scale
        mlp_input_size = (pooler_size[0] // 2) * (pooler_size[1] // 2) * hidden_size
        self.num_classes = num_classes
        if predict_scale:
            self.scale_head = self._make_mlp(
                input_size, mlp_input_size, num_classes, hidden_size, 3
            )
            self.scale_loss = nn.L1Loss()

        self.sym_head = self._make_mlp(input_size, mlp_input_size, num_classes, hidden_size, 4)
        self.sym_loss = nn.CrossEntropyLoss(torch.as_tensor([.22, .43, .78, 1.]))

    @classmethod
    def from_config(cls, cfg, shape_spec: L.ShapeSpec):
        return {
            'input_size': shape_spec.channels,
            'noc_loss_weight': cfg.MODEL.ROI_NOC_HEAD.NOC_LOSS_WEIGHT,
            'proc_loss': cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_LOSS,
            'proc_rot_weight': cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_ROTATION_WEIGHT,
            'proc_trans_weight': cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_TRANSLATION_WEIGHT,
            'eval_mask_thresh': cfg.TEST.NOC_MASK_THRESH,
            'use_deconv': cfg.MODEL.ROI_NOC_HEAD.USE_DECONV,
            'use_pixel_shuffle': cfg.MODEL.ROI_NOC_HEAD.USE_PIXEL_SHUFFLE,
            'predict_scale': cfg.MODEL.ROI_NOC_HEAD.PREDICT_SCALE,
            'pooler_size': (shape_spec.width, shape_spec.height),
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        }

    @staticmethod
    def _make_mlp(
        input_size: int,
        mlp_input_size: int,
        num_classes: int,
        hidden_size: int,
        output_size: int,
        mlp_hidden_size: int = 1024,
    ):
        return nn.Sequential(
            L.Conv2d(
                input_size,
                hidden_size,
                (5, 5),
                stride=2,
                padding=2,
                activation=nn.ReLU(True),
            ),
            nn.Flatten(),
            L.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(True),
            L.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(True),
            L.Linear(mlp_hidden_size, output_size * num_classes),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, features: Tensor, instances: list[Instances]):
        nocs, scales, syms = self.run_network(features)
        if self.training:
            return self.compute_losses(instances, nocs, scales, syms)
        else:
            return self.inference(instances, features, nocs, scales, syms)

    def inference(
        self,
        instances: list[Instances],
        features: Tensor,
        nocs: Tensor,
        scales: Optional[Tensor],
        syms: Tensor,
    ) -> list[Instances]:
        if features.numel() == 0:
            # Safe-guard against no proposals case
            for instance in instances:
                instance.pred_nocs = NOCs(tensor=torch.zeros(
                    (0, 3, features.size(-2) * 2, features.size(-1) * 2),
                    device=self.device,
                ))
                instance.pred_xy_grids = MeshGrids(tensor=torch.zeros(
                    (0, 2, features.size(-2) * 2, features.size(-1) * 2),
                    device=self.device,
                ))
                if self.predict_scale:
                    instance.pred_scales = Scales(tensor=torch.ones(0, 3, device=self.device))
                
                instance.pred_syms = torch.zeros(0, 4, dtype=torch.long, device=self.device)

        else:
            # NOTE: This can also be enabled for training if NOCs are needed outside
            lens = list(map(len, instances))
            for noc, instance in zip(nocs.split(lens), instances):
                grid = MeshGrids(instance.image_size[::-1], nocs.size(0), device=self.device)\
                    .crop_and_resize(
                        instance.pred_boxes.tensor,
                        crop_size=noc.size(-1),
                        use_interpolate=True,
                        wrap_output=False,
                )
                raw_mask = (instance.pred_masks > self.eval_mask_thresh)
                instance.pred_xy_grids = MeshGrids(tensor=(grid * raw_mask))
                instance.pred_nocs = NOCs(tensor=(noc * raw_mask))

            if self.predict_scale:
                for scale, instance in zip(scales.split(lens), instances):
                    classes = instance.pred_classes
                    scale = scale[torch.arange(classes.numel()), classes]
                    instance.pred_scales = Scales(tensor=scale)
            
            for sym, instance in zip(syms.split(lens), instances):
                classes = instance.pred_classes
                sym = sym[torch.arange(classes.numel()), classes]
                instance.pred_syms = sym.argmax(-1)

        return instances

    def run_network(self, x: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        nocs = self.output(self.shared_mlp(self.post_convs(self.upsample(self.pre_convs(x)))))
        scales = None
        if self.predict_scale:
            scales = self.scale_head(x).reshape(-1, self.num_classes, 3)
        syms = self.sym_head(x).reshape(-1, self.num_classes, 4)
        return nocs, scales, syms

    def compute_losses(
        self,
        instances: list[Instances],
        nocs: Tensor,
        scales: Optional[Tensor],
        syms: Tensor,
    ) -> dict[str, Tensor]:

        assert self.training

        losses = {}

        boxes = Boxes.cat([p.proposal_boxes for p in instances])
        gt_nocs = NOCs.cat([p.gt_nocs for p in instances])
        valid_mask = L.cat([p.gt_valid_nocs for p in instances])
        if not valid_mask.all():
            nocs = nocs[valid_mask]
            boxes = boxes[valid_mask]
            gt_nocs = gt_nocs[valid_mask]
        
        gt_classes = L.cat([p.gt_classes for p in instances])
        syms = syms[torch.arange(gt_classes.numel()), gt_classes]
        gt_syms = L.cat([p.gt_syms for p in instances])
        losses['loss_sym'] = self.sym_loss(syms, gt_syms)

        if len(boxes) == 0:  # No valid NOCs!
            loss_noc = torch.tensor(0., device=gt_nocs.device)
        else:
            gt_nocs = gt_nocs.crop_and_resize_with_grids_from_boxes(boxes, nocs.size(-1))
            # loss_masks = gt_nocs.masks()
            # diffs: Tensor = self.loss(nocs, gt_nocs.tensor) * loss_masks
            loss_noc = self._sym_aware_noc_loss(gt_syms[valid_mask], nocs, gt_nocs)
            # loss_noc = diffs.sum() / loss_masks.sum().float().clamp(1e-5)
            # self._debug(instances, loss_masks, valid_mask, boxes, gt_nocs)
        losses['loss_noc'] = self.noc_loss_weight * loss_noc

        if self.predict_scale:
            scales = scales[torch.arange(gt_classes.numel()), gt_classes]
            gt_scales = Scales.cat([p.gt_scales for p in instances]).tensor
            losses['loss_scale'] = self.scale_loss(scales, gt_scales)

        if self.proc_loss:
            losses.update(self._compute_proc_losses(instances, nocs, boxes, valid_mask, scales))

        return losses

    def _compute_proc_losses(
        self,
        instances: list[Instances],
        nocs: Tensor,
        boxes: Boxes,
        valid_mask: Tensor,
        scales: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:

        device = nocs.device
        empty_output = {
            'loss_proc_rot': torch.tensor(0., device=device),
            'loss_proc_trans': torch.tensor(0., device=device),
        }

        # Handle emptry case
        if len(boxes) == 0:
            return empty_output

        # Get gt depths
        depths = DepthPoints.cat([p.gt_depth_points for p in instances])[valid_mask]
        depths = depths.crop_and_resize_with_grids_from_boxes(boxes, nocs.size(-1))
        masks = depths.masks()

        # Remove invalids
        new_valid_mask = (masks.sum((1, 2, 3)) >= self.min_proc)
        if not new_valid_mask.any():
            return empty_output

        # Prepare Procrustes inputs
        # if scales is None:
        scales = Scales.cat([p.gt_scales for p in instances])
        # else:
        #    scales = Scales(scales)
        scales = scales[valid_mask][new_valid_mask]
        nocs = transform_grid(nocs[new_valid_mask], scales=scales)

        masks = masks[new_valid_mask]

        poses = L.cat([p.gt_poses for p in instances])[valid_mask][new_valid_mask]
        depths = depths.tensor[new_valid_mask]
        depths = transform_grid(
            depths,
            masks=masks,
            translations=Translations(poses[:, :3, 3]),
            rotations=RotationMats(poses[:, :3, :3]),
        )

        dtype = nocs.dtype
        nocs = nocs.flatten(2).transpose(1, 2).double()
        depths = depths.flatten(2).transpose(1, 2).double()
        masks = masks.flatten(1).double()

        # Perform the Procrustes alignment
        so3 = corresponding_points_alignment(nocs, depths, masks)
        rots, transes = so3.R.transpose(1, 2).to(dtype), so3.T.to(dtype)

        # Compute procrustes loss
        gt_rots = Rotations.cat([p.gt_rotations for p in instances])
        gt_rots = gt_rots[valid_mask][new_valid_mask].as_rotation_matrices().tensor
        gt_transes = Translations.cat([p.gt_translations for p in instances])
        gt_transes = gt_transes.tensor[valid_mask][new_valid_mask]

        # loss_rot = self.rot_loss(rots.flatten(1), gt_rots)
        syms = L.cat([p.gt_syms for p in instances])[valid_mask][new_valid_mask]
        loss_rot = self._sym_aware_rot_loss(syms, rots, gt_rots)
        loss_trans = self.trans_loss(transes, gt_transes)
        return {
            'loss_proc_rot': self.proc_rot_weight * loss_rot,
            'loss_proc_trans': self.proc_trans_weight * loss_trans,
        }

    def _sym_aware_rot_loss(self, gt_syms: Tensor, rots: Tensor, gt_rots: Tensor):
        rots, gt_rots = rots.flatten(1), gt_rots.flatten(1)

        '''if torch.all(gt_syms == 0):
            return self.rot_loss(rots, gt_rots)'''

        losses = []
        rots, gt_rots = rots.cpu().double(), gt_rots.cpu().double()
        gt_syms = gt_syms.cpu().tolist()
        for i, sym_type in enumerate(gt_syms):
            loss = self.rot_loss(rots[i], gt_rots[i])
            angles = self._get_angles_for_sym(sym_type)
            for angle in angles:
                rots_ = rots[i].reshape(3, 3) @ euler_angles_to_matrix(
                    torch.as_tensor([0., angle, 0.])[None].double(), 'XYZ'
                ).reshape(3, 3)
                loss = torch.minimum(loss, self.rot_loss(rots_.reshape_as(gt_rots[i]), gt_rots[i]))
            losses.append(loss)
        return torch.mean(torch.stack(losses)).float().to(self.device)

    def _sym_aware_noc_loss(self, syms: Tensor, nocs: Tensor, gt_nocs_wrap: NOCs):
        masks = gt_nocs_wrap.masks()
        gt_nocs = gt_nocs_wrap.tensor
        base_losses = self._compute_noc_losses(nocs, gt_nocs, masks)
        losses = [base_losses[syms == 0]]
        for i in range(1, 4):
            sym_mask = syms == i
            if not sym_mask.any():
                continue
            min_losses = base_losses[sym_mask]
            angles = self._get_angles_for_sym(i, 6)
            nocs_, gt_nocs_ = nocs[sym_mask], gt_nocs[sym_mask]
            masks_ = masks[sym_mask]
            for angle in angles:
                mat = euler_angles_to_matrix(
                    torch.as_tensor([0., angle, 0.], device=self.device)[None], 'XYZ'
                )
                nocs_a = transform_grid(nocs_, rotations=RotationMats(mat))
                min_losses = torch.minimum(
                    min_losses,
                    self._compute_noc_losses(nocs_a, gt_nocs_, masks_),
                )
            losses.append(min_losses)
        return torch.cat(losses).sum() / masks.sum().clamp(1e-5)

    def _compute_noc_losses(self, nocs: Tensor, gt_nocs: Tensor, masks: Tensor):
        base_diffs: Tensor = self.loss(nocs, gt_nocs) * masks
        return base_diffs.sum((1, 2, 3))

    @staticmethod
    def _get_angles_for_sym(sym_type: int, bins_for_inf: int = 36) -> list[float]:
        if sym_type == 1:
            angles = [torch.pi]
        elif sym_type == 2:
            angles = [torch.pi / 2, torch.pi, 3 * torch.pi / 2]
        elif sym_type == 3:
            angles = [x * (2 * torch.pi / bins_for_inf) for x in range(1, bins_for_inf)]
        else:
            angles = []
        return angles

    def _debug(self, instances, loss_masks, valid_mask, boxes, gt_nocs):
        gt_depths = DepthPoints.cat([p.gt_depth_points for p in instances])[valid_mask]
        gt_depths = gt_depths.crop_and_resize_with_grids_from_boxes(boxes, gt_nocs.image_size[-1])
        pose = L.cat([p.gt_poses for p in instances])[valid_mask]
        gt_depths = pose[..., :3, :3] @ gt_depths.tensor.flatten(2) + pose[..., :3, 3:4]
        gt_depths *= loss_masks.flatten(2)

        gt_translations = Translations.cat([p.gt_translations for p in instances])
        gt_rotations = Rotations.cat([p.gt_rotations for p in instances])
        gt_scales = Scales.cat([p.gt_scales for p in instances])
        gt_nocs = transform_grid(
            gt_nocs.tensor, loss_masks, gt_translations, gt_rotations, gt_scales
        )
        print((gt_nocs - gt_depths.view_as(gt_nocs))[loss_masks.expand_as(gt_nocs)].mean())
        from IPython import embed; embed()  # noqa


def build_noc_head(cfg, input_shape: L.ShapeSpec) -> nn.Module:
    return ROI_NOC_HEAD_REGISTRY.get(cfg.MODEL.ROI_NOC_HEAD.NAME)(cfg, input_shape)
