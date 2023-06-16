import copy
import os
from typing import Optional

import cv2 as cv
import numpy as np
import torch
from PIL.Image import NEAREST

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, Metadata

from nocpred.structures import DepthPoints, Depths, NOCs, Normals, Rotations, Scales, Translations


class Mapper(DatasetMapper):
    def __init__(
        self,
        cfg,
        is_train: bool,
        metadata: Optional[Metadata] = None,
        no_aug: bool = False,
        input_only: bool = False,
        # TUM-RGBD hacks!
        depth_scale: float = 1000,
        resize_and_crop: float = 0.0,
    ):
        super().__init__(cfg, is_train=is_train)

        self.debug = False

        if no_aug:
            self.augmentations = T.AugmentationList([])
        else:
            # TODO: Disable resize augmentations for now, include some later!
            self.augmentations = T.AugmentationList([
                aug for aug in self.augmentations.augs
                if not isinstance(aug, T.ResizeShortestEdge)
            ])

        self._metadata = metadata

        # Depth
        eval_alignment = cfg.TEST.IMAGE_ALIGNMENT_EVAL
        self.color_depth = cfg.INPUT.DEPTH
        self.depth_cmap = cfg.INPUT.DEPTH_CMAP
        self.color_normal = cfg.INPUT.NORMAL
        self.need_normal = self.color_normal or cfg.MODEL.ROI_NOC_HEAD.CONDITION_ON_NORMAL
        self.need_depth = (
            self.color_depth
            or cfg.MODEL.ROI_NOC_HEAD.CONDITION_ON_DEPTH
            or self.need_normal
            or self.debug
            or eval_alignment
        )

        self.depth_down_sample = cfg.INPUT.DEPTH_DOWN
        self.normal_down_sample = cfg.INPUT.NORMAL_DOWN

        # Camera and object pose
        proc_loss = cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_LOSS
        self.need_pose = not input_only and (self.debug or eval_alignment or proc_loss)
        self.need_alignments = self.debug or proc_loss
        self.need_instance_depths = self.debug or proc_loss

        if metadata is not None:
            self._cache_intrinsics()

        self.depth_scale = depth_scale
        assert 0 <= resize_and_crop <= 1, 'resize_and_crop is only for small values'
        self.resize_and_crop = resize_and_crop

    def _cache_intrinsics(self):
        # FIXME: tasks/frames stuff is annoying
        image_root = os.path.join(self._metadata.image_root, 'tasks', 'scannet_frames_25k')
        intrs = {}
        for scene_name in os.listdir(image_root):
            if 'scene' not in scene_name:
                continue
            intr_file_name = os.path.join(image_root, scene_name, 'intrinsics_depth.txt')
            with open(intr_file_name) as f:
                intr = np.array(
                    [[float(el) for el in line.strip().split()] for line in f],
                    dtype=np.float32,
                )
            intrs[scene_name] = intr[:3, :3]
        self.intrinsics_by_scene = intrs

    def __call__(self, dataset_dict: dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict['file_name'], format=self.image_format)

        if self.resize_and_crop > 0:
            h, w = image.shape[:2]
            pad = (int(w * self.resize_and_crop),  int(h * self.resize_and_crop))
            new_size = (w + pad[0], h + pad[1])
            dataset_dict['pad'] = pad
            dataset_dict['padded_image_size'] = new_size
            image = cv.resize(image, new_size, interpolation=cv.INTER_CUBIC)
            image = image[pad[1]//2:pad[1]//2+h, pad[0]//2:pad[0]//2+w]
            # image = image[]

        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        self._load_intrinsic(dataset_dict)
        self._load_pose(dataset_dict)
        self._load_depth(dataset_dict, transforms)
        self._load_normals(dataset_dict)

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
            self._load_nocs(dataset_dict, transforms)

        return dataset_dict

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict['annotations']:
            if not self.use_instance_mask:
                anno.pop('segmentation', None)
            if not self.use_keypoint:
                anno.pop('keypoints', None)

        # USER: Implement additional transformations if you have other types of data
        raw_annos = dataset_dict.pop('annotations')
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in raw_annos
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        # For matching instances with NOCs
        instances.gt_object_ids = torch.as_tensor(
            [anno['object_id'] for anno in raw_annos],
            dtype=torch.long,
        )

        # for transforms
        hflip = any(isinstance(t, T.HFlipTransform) for t in transforms)
        instances.gt_hflips = torch.as_tensor([hflip for _ in range(len(instances))])

        # Alignment labels
        # NOTE: hflip should be handled by the model!!!
        if self.need_alignments:
            transes = []
            rots = []
            scales = []
            for anno in raw_annos:
                t, q, s = anno['alignment']
                transes.append(t)
                rots.append(q)
                scales.append(s)
            instances.gt_translations = Translations(torch.as_tensor(transes))
            instances.gt_rotations = Rotations(torch.as_tensor(rots))
            instances.gt_scales = Scales(torch.as_tensor(scales))

        dataset_dict['instances'] = utils.filter_empty_instances(instances)

    def _load_intrinsic(self, dataset_dict: dict):
        if 'intrinsic' in dataset_dict:
            intr = dataset_dict['intrinsic']
        else:
            scene, _ = self._get_scene_and_file_id(dataset_dict)
            intr = self.intrinsics_by_scene[scene]
        
        if self.resize_and_crop > 0:
            intr[:2, :2] *= (1 + self.resize_and_crop)

        dataset_dict['intrinsic'] = torch.from_numpy(intr).to(dtype=torch.get_default_dtype())

    @staticmethod
    def _get_scene_and_file_id(dataset_dict: dict) -> tuple[str, int]:
        file_parts: list[str] = dataset_dict['file_name'].split('/')
        file_parts.reverse()
        scene = next(f for f in file_parts if f.startswith('scene'))
        file_id = int(file_parts[0].replace('.jpg', ''))
        return scene, file_id

    def _load_depth(self, dataset_dict: dict, transforms: T.TransformList):
        if not self.need_depth:
            return

        if 'depth_file_name' in dataset_dict:
            depth_file_name: str = dataset_dict.pop('depth_file_name')
        else:
            file_name: str = dataset_dict['file_name']
            depth_file_name = file_name.replace('color', 'depth').replace('.jpg', '.png')
        
        if 'padded_image_size' in dataset_dict:
            px, py = dataset_dict['pad']
            w, h = dataset_dict['width'], dataset_dict['height']
            dataset_dict['depth'] = depth = Depths.from_file(
                depth_file_name,
                lambda x: cv.resize(
                    transforms.apply_image(x),
                    dataset_dict['padded_image_size'],
                    interpolation=cv.INTER_NEAREST)[py//2:py//2+h, px//2:px//2+w],
                self.depth_scale,
            )
        else:
            dataset_dict['depth'] = depth = Depths.from_file(
                depth_file_name,
                transforms.apply_image,
                self.depth_scale,
            )
        if self.color_depth:
            bgr = self.image_format == 'BGR'
            dataset_dict['depth_image'] = depth.to_color(
                bgr=bgr, color_map=self.depth_cmap, down_sample=self.depth_down_sample
            ).tensor[0].contiguous()
            # if any(isinstance(t, T.HFlipTransform) for t in transforms):
            #    self._save_image_for_debug(dataset_dict, 'depth_image')

    def _load_normals(self, dataset_dict: dict):
        if not self.need_normal:
            return
        dataset_dict['normals'] = normals = Normals.from_depths(
            dataset_dict['depth'], down_sample=self.normal_down_sample
        )
        if self.color_normal:
            bgr = self.image_format == 'BGR'
            dataset_dict['normal_image'] = normals.to_color(bgr=bgr).tensor[0].contiguous()

    def _save_image_for_debug(self, dataset_dict: dict, field_name: str):
        scene, file_id = self._get_scene_and_file_id(dataset_dict)
        cv.imwrite(
            '{}_{}_{}.jpg'.format(scene, file_id, field_name),
            dataset_dict[field_name].permute(1, 2, 0).contiguous().numpy(),
        )
        exit()

    def _load_pose(self, dataset_dict: dict):
        if not self.need_pose:
            return
        file_name: str = dataset_dict['file_name']
        file_name = file_name.replace('color', 'pose').replace('.jpg', '.txt')
        with open(file_name) as f:
            pose = torch.tensor([[float(el) for el in line.strip().split()] for line in f])
        dataset_dict['pose'] = pose

    def _load_nocs(self, dataset_dict: dict, transforms: T.Transform):
        file_name: str = dataset_dict['file_name']
        noc_file_name = file_name\
            .replace(self._metadata.image_root, self._metadata.rendering_root)\
            .replace('/tasks/scannet_frames_25k', '')\
            .replace('color', 'noc')\
            .replace('.jpg', '.png')

        id_file_name = noc_file_name.replace('noc', 'id')
        instance_img = cv.imread(id_file_name, cv.IMREAD_GRAYSCALE)
        assert instance_img is not None, '{} cannot be loaded'.format(id_file_name)
        instance_img = transforms.apply_image(instance_img)

        instances = dataset_dict['instances']
        object_ids = instances.gt_object_ids.tolist()

        nocs, valids = NOCs.from_file(noc_file_name, transforms).unroll_instances_with_given_ids(
            instance_img, object_ids
        )
        instances.gt_nocs = nocs
        instances.gt_valid_nocs = torch.as_tensor(valids, dtype=torch.bool)

        if self.need_instance_depths:
            self._set_instance_depths_and_poses(dataset_dict, instance_img, object_ids)

    @staticmethod
    def _set_instance_depths_and_poses(dataset_dict, instance_img, object_ids):
        instances = dataset_dict['instances']
        depth = dataset_dict['depth']
        depth: DepthPoints = depth.back_project(dataset_dict['intrinsic'])
        instances.gt_depth_points = depth.unroll_instances_with_given_ids(
            instance_img, object_ids, ret_valids=False
        )
        instances.gt_poses = torch.stack([dataset_dict['pose'] for _ in range(len(instances))])
