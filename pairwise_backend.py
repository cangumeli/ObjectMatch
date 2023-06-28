import copy
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import cv2 as cv
import numpy as np
import open3d as o3d
import PIL.Image as Image
import trimesh
from scipy.optimize import linear_sum_assignment

import torch
from detectron2.structures import Instances
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

from nocpred.config import nocpred_config
from nocpred.engine import Predictor as NOCPredictor
from objectid.predictor import Predictor as ObjectEmbedder
from optim.common import make_M_from_tqs, rotate_x
from optim.solver import (
    batch_instance_data,
    distance_filter,
    filter_and_initalize,
    frame_energy,
    InstanceDatum,
    overlap_energy,
    refine_icp,
)


@dataclass(frozen=True)
class NOCPredConfig:
    model_file: str
    config_file: str
    score_thresh: float = .5  # 5
    noc_mask_thresh: float = .4  # .4
    nms_thresh: float = .3  # .3
    non_sym: bool = True
    filtered_init: bool = True
    min_nocs: int = 20
    proc_thesh: float = .2

    @cached_property
    def d2_cfg(self):
        cfg = nocpred_config(self.config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.TEST.NOC_MASK_THRESH = self.noc_mask_thresh
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_thresh
        return cfg


@dataclass(frozen=True)
class ObjectIDConfig:
    folder: str
    thresh: float = .03
    best_effort: tuple[float] = (.05, .1, .15, .2, .3, .5, 1.)
    top: int = 1
    scale_ratio_thresh: float = 1.5
    select_max_nocs: bool = True


@dataclass(frozen=True)
class KPConfig:
    min_points: int = 4
    proc_thresh: float = .2


@dataclass(frozen=True)
class OptimConfig:
    num_iters: int = 30
    dumping_factor: float = 1e-5
    dumping_coef: float = 10.
    kp_weight: float = 1.
    obj_weight: float = 0.01
    use_lm: bool = True
    outlier_iter: int = 15
    outlier_thresh: float = .15
    verbose: bool = True
    icp_voxel_size: float = 0.03  # 3
    icp_thresh: float = 0.05


@dataclass(frozen=True)
class GNOutput:
    cam_transes: torch.Tensor
    cam_angles: torch.Tensor
    obj_transes: Optional[torch.Tensor] = None
    obj_angles: Optional[torch.Tensor] = None
    obj_scales: Optional[torch.Tensor] = None
    success: bool = True
    loss_history: Optional[list[float]] = None

    @cached_property
    def cam_pose(self):
        pred_pose = np.eye(4)
        cam_angles = self.cam_angles.cpu()
        cam_transes = self.cam_transes.cpu()
        pred_pose[:3, :3] = euler_angles_to_matrix(cam_angles, 'XYZ').squeeze().cpu().numpy()
        pred_pose[:3, 3] = cam_transes.cpu().squeeze().numpy()
        return pred_pose

    @cached_property
    def has_obj(self):
        return bool(self.obj_transes.numel())

    def obj_poses(self, div_scale=1.25) -> list[np.ndarray]:
        if not self.has_obj:
            return []

        result = []
        angles = matrix_to_quaternion(euler_angles_to_matrix(self.obj_angles.cpu(), 'XYZ'))
        # Canonical space is normalized between [-.4, +.4]
        scales = self.obj_scales.cpu().abs() / div_scale
        for t, q, s in zip(self.obj_transes.cpu(), angles, scales):
            t = t.flatten().tolist()
            q = q.flatten().tolist()
            s = s.flatten().tolist()
            result.append(make_M_from_tqs(t, q, s))
        return result


@dataclass
class OptimExtraOutputs:
    kp_locations: list[tuple[tuple[float, float], tuple[float, float]]]
    kp_mask: np.ndarray
    obj_instances: tuple[Instances, Instances]


class PairwiseSolver:
    def __init__(
        self,
        kp_cfg: KPConfig,
        nocpred_cfg: NOCPredConfig,
        objectid_cfg: ObjectIDConfig,
        optim_cfg: OptimConfig,
        print_fn = print,
    ):
        self.kp_cfg = kp_cfg
        self.objectid_cfg = objectid_cfg
        self.nocpred_cfg = nocpred_cfg
        self.optim_cfg = optim_cfg
        self.print_fn = print_fn

        self._build_networks()

    def _build_networks(self):
        self.print_fn('Building networks...')
        self.noc_pred = NOCPredictor(self.nocpred_cfg.d2_cfg, self.nocpred_cfg.model_file)
        self.embed_pred = ObjectEmbedder(self.objectid_cfg.folder)

    def load_record(self, *args, **kwargs):
        record = self.noc_pred.load_record(*args, **kwargs)
        record['depth_points'] = record['depth'].back_project(record['intrinsic'][:3, :3])
        return record

    @property
    def device(self):
        return next(self.noc_pred.model.parameters()).device

    def __call__(
        self,
        record0: dict,
        record1: dict,
        match_data: dict[str, np.ndarray],
        ret_extra_outputs: bool = False,
    ):
        # Process keypoints
        overlaps, kp_locations, kp_mask = self._prep_kp(record0, record1, match_data)
        no_kp = not np.any(kp_mask)

        # Find and process objects
        res0 = self._run_nocpred(record0)
        res1 = self._run_nocpred(record1)
        res0, res1 = self._run_objectid(record0, record1, res0, res1, best_effort=no_kp)
        instance_data = self._prep_instances(res0, res1)

        # Run global GN optimization
        output_gn = self._run_gn(overlaps, instance_data, num_objects=len(res0))

        # Run local ICP refinement
        output = self._run_icp(record0, record1, output_gn.cam_pose)

        # Both the final camera pose and global optim output are returned
        if ret_extra_outputs:
            extras = OptimExtraOutputs(kp_locations, kp_mask, (res0, res1))
            return output, output_gn, extras
        else:
            return output, output_gn

    def _prep_kp(self, record0: dict, record1: dict, match_data: dict[str, np.ndarray]):
        matches = match_data['matches']
        kpts0 = match_data['keypoints0']
        kpts1 = match_data['keypoints1']

        ind0 = np.argwhere(matches > 0).ravel()
        ind1 = matches[matches > 0]

        # get back-projected depths
        depth0 = record0['depth_points'].tensor.squeeze(0).numpy()
        depth1 = record1['depth_points'].tensor.squeeze(0).numpy()

        locations = []
        depths = []
        for kp0, kp1 in zip(kpts0[ind0], kpts1[ind1]):
            # print(kp0, kp1)
            c0, r0 = map(int, kp0)
            c1, r1 = map(int, kp1)
            d0 = depth0[:, r0, c0]
            d1 = depth1[:, r1, c1]
            if np.all(d0 == 0) or np.all(d1 == 0):
                continue
            locations.append(((c0, r0), (c1, r1)))
            depths.append((d0, d1))

        pts0 = np.array([])
        pts1 = np.array([])
        mask = np.zeros(len(depths), dtype=np.bool_)
        if len(depths) > self.kp_cfg.min_points:
            pts0 = np.stack([d0 for d0, _ in depths])
            pts1 = np.stack([d1 for _, d1 in depths])
            mask, _ = distance_filter(
                pts0,
                pts1,
                np.ones(pts0.shape[0], dtype=np.bool_),
                thresh=self.kp_cfg.proc_thresh,
            )
            # if mask is not None and mask.size and np.sum(mask) >= 5:
            mask = mask.ravel()
            # import pdb; pdb.set_trace()
            pts0 = pts0[mask]
            pts1 = pts1[mask]

        if pts0.shape[0] <= self.kp_cfg.min_points:
            pts0 = np.array([])
            pts1 = np.array([])
            mask = np.zeros_like(mask)

        scores = torch.ones(pts0.shape[0]) / pts0.shape[0]

        overlaps = [(
            0,
            1,
            torch.from_numpy(pts0).float().to(self.device),
            torch.from_numpy(pts1).float().to(self.device),
            scores.to(self.device),
        )]

        return overlaps, locations, mask

    def _run_nocpred(self, record: dict):
        # Run the actual forward pass
        result = self.noc_pred(record)['instances']

        # prep depths
        if len(result):
            depth = record['depth_points'].to(result.pred_masks.device)
            depth = depth.repeat(len(result))
            depth_points = depth.crop_and_resize_with_grids_from_boxes(
                result.pred_boxes, crop_size=result.pred_nocs.image_size[-1]
            )
            result.pred_depths = depth_points

            if self.nocpred_cfg.filtered_init:
                converged, new_masks = filter_and_initalize(
                    result.pred_nocs.tensor,
                    result.pred_depths.tensor,
                    result.pred_nocs.masks() & result.pred_depths.masks(),
                    result.pred_scales.tensor,
                    self.nocpred_cfg.proc_thesh,
                    min_mask=self.nocpred_cfg.min_nocs,
                )[-2:]
                result.pred_optim_masks = new_masks
                result = result[converged]
            else:
                result.pred_optim_masks = result.pred_nocs.masks() & result.pred_depths.masks()

            if self.nocpred_cfg.non_sym:
                result = result[result.pred_syms == 0]

        return result

    def _run_objectid(self, record0, record1, res0, res1, best_effort):
        if not len(res0) or not len(res1):
            return res0, res1

        image0 = Image.fromarray(record0['image'].permute(1, 2, 0).numpy()[..., ::-1])
        image1 = Image.fromarray(record1['image'].permute(1, 2, 0).numpy()[..., ::-1])

        # Compute embeddings
        scores0 = self.embed_pred(
            image0,
            depth=record0['depth'].tensor.squeeze().numpy(),
            boxes=res0.pred_boxes.tensor.cpu().tolist(),
            masks=res0.pred_masks.cpu().numpy(),
            classes=res0.pred_classes.cpu().tolist(),
        )
        scores1 = self.embed_pred(
            image1,
            depth=record1['depth'].tensor.squeeze().numpy(),
            boxes=res1.pred_boxes.tensor.cpu().tolist(),
            masks=res1.pred_masks.cpu().numpy(),
            classes=res1.pred_classes.cpu().tolist(),
        )

        # Compute the distance table
        dists = (scores0[:, None] - scores1[None]).square_().mean(-1)
        dists[res0.pred_classes[:, None] != res1.pred_classes[None]] = 1000
        dists[res0.pred_syms[:, None] != res1.pred_syms[None]] = 1000
        dists = dists.cpu().numpy()

        # Apply scale filter
        ratios = res0.pred_scales.tensor[:, None] / res1.pred_scales.tensor[None]
        ratios = ratios.max(-1).values
        ratios[ratios < 1] = 1 / ratios[ratios < 1]
        dists[(ratios >= self.objectid_cfg.scale_ratio_thresh).cpu().numpy()] = 1000

        # Associate ids
        idx0, idx1 = linear_sum_assignment(dists)
        match_dists: np.ndarray = dists[idx0, idx1]
        best = np.argsort(match_dists)
        match_dists = match_dists[best]
        idx0, idx1 = idx0[best], idx1[best]

        # Do the actual filtering & selection
        thresholds = (self.objectid_cfg.thresh,)
        if best_effort:
            thresholds += self.objectid_cfg.best_effort

        def select_with_thresh(idx0, idx1, res0, res1, thresh):
            top = self.objectid_cfg.top
            flt = match_dists < thresh
            idx0, idx1 = idx0[flt], idx1[flt]
            if top == 1 and len(idx0) and self.objectid_cfg.select_max_nocs:
                res0, res1 = res0[idx0], res1[idx1]
                counts = torch.minimum(
                    res0.pred_optim_masks.flatten(1).sum(-1),
                    res1.pred_optim_masks.flatten(1).sum(-1),
                )
                idx = counts.argmax().item()
                res0, res1 = res0[idx:idx+1], res1[idx:idx+1]
            else:
                res0, res1 = res0[idx0][:top], res1[idx1][:top]
            return res0, res1

        for thresh in thresholds:
            res0_, res1_ = select_with_thresh(idx0, idx1, res0, res1, thresh)
            if len(res0_):
                break

        return res0_, res1_

    def _prep_instances(self, res0, res1):
        instance_data = None

        if len(res0) and len(res1):
            outputs = [
                {
                    'ids': list(range(1, 1 + len(res0))),
                    'nocs': res0.pred_nocs.tensor,
                    'classes': res0.pred_classes.cpu(),
                    'masks': res0.pred_optim_masks,
                    'depths': res0.pred_depths.tensor,
                },
                {
                    'ids': list(range(1, 1 + len(res1))),
                    'nocs': res1.pred_nocs.tensor,
                    'classes': res1.pred_classes.cpu(),
                    'masks': res1.pred_optim_masks,
                    'depths': res1.pred_depths.tensor,
                },
            ]
            instance_data = [InstanceDatum(**output) for output in outputs]
            instance_data = batch_instance_data(instance_data)

        return instance_data

    def _run_gn(self, overlaps, instance_data, num_objects) -> GNOutput:
        device = self.device

        cam_angles = torch.zeros(1, 3, device=device)
        cam_transes = torch.zeros(1, 3, device=device)
        obj_angles = torch.zeros(num_objects, 3, device=device)
        obj_transes = torch.zeros(num_objects, 3, device=device)
        obj_scales = torch.ones(num_objects, 3, device=device)
    
        loss_history = []
        cfg = self.optim_cfg
        dumping_factor = cfg.dumping_factor

        for i in range(cfg.num_iters):
            # Compute the keypoint jacobian
            try:
                jacob_c, res_c, _ = overlap_energy(overlaps, cam_angles, cam_transes)
            except RuntimeError as e:  # No keypoints
                if i == 0 and cfg.verbose:
                    self.print_fn(e)
                jacob_c, res_c = torch.zeros(0, 6), torch.tensor([])

            if cfg.kp_weight != 1:
                res_c *= cfg.kp_weight
                jacob_c *= cfg.kp_weight
    
            # Compute object jacobian
            jacob, res = jacob_c, res_c
            if instance_data is not None:
                jacob_f, res_f = frame_energy(
                    instance_data,
                    cam_angles.to(device),
                    cam_transes.to(device),
                    obj_angles,
                    obj_transes,
                    obj_scales,
                )
                jacob_f *= cfg.obj_weight
                res_f *= cfg.obj_weight

                if jacob_c.numel():
                    jacob_c = torch.cat([
                        jacob_c,
                        torch.zeros(
                            jacob_c.size(0),
                            jacob_f.size(1) - jacob_c.size(1),
                            device=jacob_c.device,
                        )
                    ], dim=1)
                    jacob = torch.cat([jacob_c.to(device), jacob_f])
                    res = torch.cat([res_c.to(device), res_f])
                else:
                    jacob, res = jacob_f, res_f

            # Perform outlier removal
            if i > cfg.outlier_iter:
                res_ = res.reshape(-1, 3).clone()
                res_[res_c.numel():] *= (1 / cfg.obj_weight)

                norm = res_.norm(dim=-1)
                flt = norm < cfg.outlier_thresh
                flt = flt.repeat_interleave(3)
                jacob = jacob[flt]
                res = res[flt]

            # Run the optimization step
            # dumping_factor = cfg.dumping_factor
            try:
                jtj = jacob.t() @ jacob
                jtf = jacob.t() @ res
                if cfg.use_lm:
                    jtj = jtj + dumping_factor * torch.eye(jtj.size(0), device=jtj.device) * jtj
                step = torch.linalg.solve(2 * jtj, jtf).flatten()
            except RuntimeError as e:
                if cfg.verbose:
                    self.print_fn(e)
                break
            num_views = 2
            if step.numel() > 6:
                step_cam, step_obj = step.split([(num_views - 1) * 6, num_objects * 9])
            else:
                step_cam = step

            cam_angles = cam_angles - torch.cat([s[:3] for s in step_cam.chunk(num_views - 1)])\
                .view_as(cam_angles)
            cam_transes = cam_transes - torch.cat([s[3:6] for s in step_cam.chunk(num_views - 1)])\
                .view_as(cam_transes)

            if step.numel() > 6:
                obj_angles = obj_angles - torch.cat([s[:3] for s in step_obj.chunk(num_objects)])\
                    .view_as(obj_angles)
                obj_transes = obj_transes - torch.cat([s[3:6] for s in step_obj.chunk(num_objects)])\
                    .view_as(obj_transes)
                obj_scales = obj_scales - torch.cat([s[6:] for s in step_obj.chunk(num_objects)])\
                    .view_as(obj_scales)
                if cfg.verbose:
                    self.print_fn(res_c.square().sum().item(), res_f.square().sum().item())
            elif cfg.verbose:
                self.print_fn(res.square().sum().item())

            # Update loss history and LM dumping factor
            loss_history.append(res.square().sum().item())

            if len(loss_history) >= 2 and cfg.use_lm:
                dumping_factor *= (
                    1 / cfg.dumping_coef
                    if loss_history[-1] < loss_history[-2]
                    else cfg.dumping_coef
                )

        success = i > 0
        if cfg.verbose:
            self.print_fn('GN solver {}'.format('succeeded :)!' if success else 'failed :(!'))

        return GNOutput(
            cam_transes,
            cam_angles,
            obj_transes,
            obj_angles,
            obj_scales,
            success,
            loss_history,
        )

    def _run_icp(self, record0, record1, init_pose: np.ndarray) -> np.ndarray:
        # Prepare point clouds
        pcd0 = record0['depth_points'].as_point_clouds(False)[0].numpy()
        pcd1 = record1['depth_points'].as_point_clouds(False)[0].numpy()

        # Run the actual icp
        return refine_icp(
            init_pose,
            pcd0,
            pcd1,
            voxel_size=self.optim_cfg.icp_voxel_size,
            thresh=self.optim_cfg.icp_thresh,
        )


class PairwiseVisualizer:
    def __init__(self, vis_output_dir: str = 'vis_pairs', print_fn=print):
        self.vis_output_dir = vis_output_dir
        self.print_fn = print_fn

    def __call__(
        self,
        vis_name: str,
        record0: dict,
        record1: dict,
        pred_pose: np.ndarray,
        gn_output: GNOutput,
        extra_outputs: Optional[OptimExtraOutputs] = None,
        asset_root: str = './assets/',
        depth_trunc: float = 6.,
        filter_depth: bool = False,
        clean_mesh: bool = False,
    ):
        logging.getLogger('trimesh').setLevel(logging.ERROR)
        output_dir = os.path.join(self.vis_output_dir, vis_name)
        os.makedirs(output_dir, exist_ok=True)

        self._dump_input(output_dir, record0, record1)
        self._dump_registration(
            output_dir, record0, record1, pred_pose, depth_trunc, filter_depth, clean_mesh
        )
        self._dump_cameras(output_dir, pred_pose, asset_root)
        self._dump_objects(output_dir, gn_output, asset_root)
        if extra_outputs is not None:
            self._dump_extras(output_dir, record0, record1, extra_outputs)

    def _dump_input(self, output_dir, record0, record1):
         # Visualize and dump the image inputs
        out_file = os.path.join(output_dir, 'input.jpg')
        self.print_fn(f'Visualizing RGB-D inputs to {out_file}...')
        image0 = record0['image'].permute(1, 2, 0).numpy()[..., ::-1]
        image1 = record1['image'].permute(1, 2, 0).numpy()[..., ::-1]
        depth_jet0 = record0['depth_image'].permute(1, 2, 0).numpy()[..., ::-1]
        depth_jet1 = record1['depth_image'].permute(1, 2, 0).numpy()[..., ::-1]

        normal0 = record0['normal_image'].permute(1, 2, 0).numpy()[..., ::-1]
        normal1 = record1['normal_image'].permute(1, 2, 0).numpy()[..., ::-1]
        normal0 = cv.resize(normal0, image0.shape[:2][::-1], interpolation=cv.INTER_NEAREST)
        normal1 = cv.resize(normal1, image0.shape[:2][::-1], interpolation=cv.INTER_NEAREST)

        image_grid = np.vstack([
            np.hstack([image0, depth_jet0, normal0]),
            np.hstack([image1, depth_jet1, normal1]),
        ])
        Image.fromarray(image_grid).save(out_file)
    
    def _dump_registration(
        self,
        output_dir,
        record0,
        record1,
        pred_pose,
        depth_trunc=6.,
        filter_depth: bool = True,
        clean_mesh: bool = True,
    ):
        self.print_fn('Registering rgbd images...')
        image0 = np.ascontiguousarray(record0['image'].permute(1, 2, 0).numpy()[..., ::-1])
        image1 = np.ascontiguousarray(record1['image'].permute(1, 2, 0).numpy()[..., ::-1])

        intr0 = record0['intrinsic'].numpy()
        intr0 = o3d.camera.PinholeCameraIntrinsic(
            *image0.shape[:2][::-1], intr0[0, 0], intr0[1, 1], intr0[0, 2], intr0[1, 2]
        )
        intr1 = record1['intrinsic'].numpy()
        intr1 = o3d.camera.PinholeCameraIntrinsic(
            *image1.shape[:2][::-1], intr1[0, 0], intr1[1, 1], intr1[0, 2], intr1[1, 2]
        )

        color0 = o3d.geometry.Image(np.ascontiguousarray(image0))
        color1 = o3d.geometry.Image(np.ascontiguousarray(image1))
        scale0 = record0['depth'].scale
        scale1 = record1['depth'].scale
        depth0 = o3d.geometry.Image(record0['depth'].encode('uint16').squeeze())
        depth1 = o3d.geometry.Image(record1['depth'].encode('uint16').squeeze())
        image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color0, depth0,
            depth_scale=scale0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
        )
        image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color1, depth1,
            depth_scale=scale1, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
        )

        if filter_depth:
            for depth in map(np.asarray, (image0.depth, image1.depth)):
                # TODO: Make this configurable
                depth_ = cv.bilateralFilter(depth, 9, 75, 75, borderType=cv.BORDER_DEFAULT)
                np.copyto(depth, depth_)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            # voxel_length=3.0 / 512.0,
            voxel_length=0.01,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            # origin=1.5 * np.ones(3)[..., None],
        )
        volume.integrate(image0, intr0, np.eye(4))
        volume.integrate(image1, intr1, np.linalg.inv(pred_pose))

        mesh = volume.extract_triangle_mesh()
        mesh.transform(rotate_x(180))

        if clean_mesh:
            mesh = self._clean_mesh(mesh)

        out_dir = os.path.join(output_dir, 'mesh')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'registration.ply')
        self.print_fn(f'Writing registered mesh to {out_file}...')
        o3d.io.write_triangle_mesh(out_file, mesh)

    def _clean_mesh(self, mesh):
        self.print_fn('Cleaning the registered mesh...')
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            mesh_0 = copy.deepcopy(mesh)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < 1000
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
        return mesh_0

    def _dump_cameras(self, output_dir, pred_pose, asset_root):
        self.print_fn('Visualizing cameras...')
        camera = trimesh.load(os.path.join(asset_root, 'camera.obj'), force='mesh')
        camera.apply_scale(.5)
        camera.apply_transform(rotate_x(180))

        out_dir = os.path.join(output_dir, 'mesh')
        
        cam0 = camera.copy().as_open3d
        cam0.paint_uniform_color([0., 1., 0.])
        cam0.transform(rotate_x(180))
        out_file0 = os.path.join(out_dir, 'cam0.ply')
        self.print_fn(f'Dumping souce camera to {out_file0}')
        o3d.io.write_triangle_mesh(out_file0, cam0)

        cam1 = camera.copy()
        cam1.apply_transform(pred_pose)
        cam1 = cam1.as_open3d
        cam1.paint_uniform_color([0., .8, .8])
        cam1.transform(rotate_x(180))
        out_file1 = os.path.join(out_dir, 'cam1.ply')
        self.print_fn(f'Dumping registered camera to {out_file1}')
        o3d.io.write_triangle_mesh(out_file1, cam1)

    def _dump_objects(self, output_dir, gn_output: GNOutput, asset_root: str):
        self.print_fn('Visualizing object bounding boxes...')
        if not gn_output.has_obj:
            return

        out_dir = os.path.join(output_dir, 'mesh')
        box = o3d.io.read_triangle_mesh(os.path.join(asset_root, 'bbox.ply'))
        box.paint_uniform_color([.5, .5, .8])

        obj_poses = gn_output.obj_poses(div_scale=2.5)  # account for NOC space + bbox scale
        for k, pose in enumerate(obj_poses):
            out_file = os.path.join(out_dir, f'box{k}.ply')
            box_ = type(box)(box)  # copy
            box_.transform(pose)
            box_.transform(rotate_x(180))
            self.print_fn(f'Writing a bbox to {out_file}...')
            o3d.io.write_triangle_mesh(out_file, box_)

    def _dump_extras(self, output_dir, record0, record1, extra_outputs: OptimExtraOutputs):
        image0 = record0['image'].permute(1, 2, 0).numpy()[..., ::-1]
        image1 = record1['image'].permute(1, 2, 0).numpy()[..., ::-1]
        self._dump_nocs(output_dir, image0, image1, extra_outputs)
        self._dump_kps(output_dir, image0, image1, extra_outputs)

    def _dump_nocs(
        self,
        output_dir,
        image0: np.ndarray,
        image1: np.ndarray,
        extra_outputs: OptimExtraOutputs,
    ):
        self.print_fn('Visualizing NOCs...')
        res0, res1 = extra_outputs.obj_instances
        if not len(res0) or not len(res1):
            self.print_fn('No valid NOCs!')
            return
        image0 = image0.astype('float32').copy()
        image1 = image1.astype('float32').copy()
        for res, img in zip((res0, res1), (image0, image1)):
            img_ = np.zeros_like(img)
            for xy, noc, box, mask in zip(
                res.pred_xy_grids.tensor.cpu().numpy(),
                res.pred_nocs.tensor.cpu().numpy(),
                res.pred_boxes.tensor.cpu().numpy(),
                res.pred_masks.cpu().numpy(),
            ):
                noc = 255 * (.5 + noc.reshape(3, -1).T)
                xy = xy.reshape(2, -1).T.astype(np.int64).tolist()
                x0, y0, x1, y1 = box.tolist()
                offset_x = max(int((x1 - x0) / 32), 1)
                offset_y = max(int((y1 - y0) / 32), 1)
                for (x, y), n in zip(xy, noc):
                    if np.allclose(n, 0):
                        continue
                    img_[y-offset_y:y+offset_y+1, x-offset_x:x+offset_x+1] = n
                img_ = cv.medianBlur(img_.astype('uint8'), 23)
                mask = (img_ != 0).any(-1) & mask
                img[mask] = .2 * img[mask] + .8 * img_[mask]
        image = np.hstack([image0, image1]).astype('uint8')
        out_file = os.path.join(output_dir, 'noc.jpg')
        self.print_fn(f'Dumping nocs to {out_file}')
        Image.fromarray(image).save(out_file)

    def _dump_kps(
        self,
        output_dir,
        image0: np.ndarray,
        image1: np.ndarray,
        extra_outputs: OptimExtraOutputs,
    ):
        self.print_fn('Visualizing keypoints...')
        mask = extra_outputs.kp_mask
        locations = extra_outputs.kp_locations

        if not mask.size:
            self.print_fn('No valid keypoint matches!')
            return

        img = np.ascontiguousarray(np.hstack([image0, image1]))
        for ((c0, r0), (c1, r1)), m in zip(locations, mask.tolist()):
            if m:  # inlier
                color = (0, 220, 0)
            else:
                color = (220, 0, 0)
            cv.line(img, (c0, r0), (image0.shape[1] + c1, r1), color, thickness=2)
        
        out_file = os.path.join(output_dir, 'kp.jpg')
        self.print_fn(f'Dumping keypoint matches to {out_file}')
        Image.fromarray(img).save(out_file)
