import os
from typing import Iterable, Optional

import numpy as np
import open3d as o3d
import PIL.Image as Image
import torch
from matplotlib import pyplot as plt

from optim.common import load_depth, load_depth_cached, load_matrix
from optim.roca_ops import back_project
from optim.solver import test


def preprocess_point_cloud(pcd, voxel_size, feature=True):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    if feature:
        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    else:
        pcd_fpfh = None
    return pcd_down, pcd_fpfh


def load_point_clouds(
    data_dir: str,
    scene: str,
    trajectory: Iterable[int],
    voxel_size: float = 0.05,
    debug: bool = False,
    feature: bool = True,
    cached: bool = False,
):
    pose_files = [os.path.join(data_dir, scene, 'pose', '{}.txt'.format(i)) for i in trajectory]
    poses = [load_matrix(f) for f in pose_files]
    to_1st_camera = np.linalg.inv(poses[0])
    poses = [to_1st_camera @ pose for pose in poses]
    # print('Loading depths...')
    depth_files = [os.path.join(data_dir, scene, 'depth', '{}.png'.format(i)) for i in trajectory]
    depths = [(load_depth(f, 'float32') if not cached else load_depth_cached(f)) for f in depth_files]
    if debug:
        colored_depths = [plt.get_cmap('jet_r')(d / d.max())[:, :, :3] for d in depths]

    depths = torch.stack([torch.from_numpy(d) for d in depths]).unsqueeze(1)
    y, x = torch.meshgrid(
        torch.linspace(0, 479, 480), torch.linspace(0, 639, 640)
    )
    xy_grid = torch.stack([x, y])
    xy_grid = xy_grid[None].expand(len(trajectory), 2, 480, 640)

    try:
        intr = load_matrix(os.path.join(data_dir, scene, 'intrinsic_depth.txt'))
    except FileNotFoundError:
        intr = load_matrix(os.path.join(data_dir, scene, 'intrinsic/intrinsic_depth.txt'))
    intr = torch.as_tensor(intr[:3, :3]).float()
    depth_points = back_project(xy_grid, depths, intr)

    # print('Loading colors...')
    color_files = [os.path.join(data_dir, scene, 'color', '{}.jpg'.format(i)) for i in trajectory]
    images = [Image.open(f) for f in color_files]
    colors = [np.asarray(img).astype(np.float32) / 255. for img in images]

    if debug:
        pad = np.ones((colors[0].shape[0], 10, 3), dtype=colors[0].dtype)
        colors_to_show = [np.hstack([pad, color, pad]) for color in colors]
        colors_to_show = np.hstack(colors_to_show)
        depths_to_show = np.hstack([np.hstack([pad, d, pad]) for d in colored_depths])
        colors_to_show = np.vstack([colors_to_show, depths_to_show])
        # o3d.visualization.draw_geometries([o3d.geometry.Image(np.hstack(colors_to_show))])

    # print('Preparing point clouds...')
    pcds = []
    for points, color in zip(depth_points, colors):
        mask = points[2, :, :] > 0
        points = points.permute(1, 2, 0).contiguous().numpy()[mask.numpy()]
        color = color[mask.numpy()]
        pcd = o3d.geometry.PointCloud()
        # import pdb; pdb.set_trace()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(color)
        pcds.append(pcd)

    pcd_feat = [preprocess_point_cloud(pcd, voxel_size, feature=feature) for pcd in pcds]

    if debug:
        return pcd_feat, poses, colors_to_show, colors
    return pcd_feat, poses


def depth_to_pcd(points, color: np.ndarray, voxel_size: float = 0.05, feature=True):
    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(np.ascontiguousarray(points))
    if points.size(0) != 3:
        points = points.permute(2, 0, 1)
    points = points.cpu()
    mask = points[2, :, :] > 0
    points = points.permute(1, 2, 0).contiguous().numpy()[mask.numpy()]
    color = color[mask.numpy()]

    # import pdb; pdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color)

    return preprocess_point_cloud(pcd, voxel_size, feature=feature)


def pairwise_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size,
    fast=False,
    icp_only=False,
    ret_result=False,
    throw_if_icp=False,
    icp_thresh: float = 0,  # 0.5,
    init_pose=np.eye(4),
):
    # print("Executing global pairwise registration...")
    distance_threshold = voxel_size * 1.5
    icp_distance_threshold = voxel_size * 0.3
    
    if icp_only:
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold,
            # np.eye(4),
            init_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_down, target_down, distance_threshold,
            result.transformation,
        )

        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, icp_distance_threshold,
            # np.eye(4),
            result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_down, target_down, icp_distance_threshold,
            result.transformation,
        )

        if ret_result:
            return result, info
        return result.transformation, info

    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    if fast:
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
             source_down, target_down, source_fpfh, target_fpfh
        )
    else:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
             source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            # o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            seed=2,
        )
        # from IPython import embed; embed()

    fitness = result.fitness
    if result.fitness < icp_thresh:
        if throw_if_icp:
            raise RuntimeError('ICP')
        print('ICP...')
        # raise RuntimeError('ICP')
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, 3 * distance_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        init_transform = result.transformation
    else:
        init_transform = result.transformation

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, icp_distance_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, icp_distance_threshold,
        result.transformation,
    )
    if ret_result:
        return result.transformation, info, fitness
    else:
        return result.transformation, info


CACHE = {}


def solve_fpfh_pair(
    data_dir: str,
    scene: str,
    pair: tuple[int, int],
    voxel_size: float = 0.05,
    fast: bool = False,
    icp_only: bool = False,
    get_info: bool = False,
    init_pose: Optional[np.ndarray] = None,
    icp_thresh: float = 0.0,
    throw_if_icp: bool = False,
):
    [(src_down, src_feat), (dst_down, dst_feat)], poses = load_point_clouds(
        data_dir, scene, pair, voxel_size
    )

    result, info = pairwise_registration(
        src_down,
        dst_down,
        src_feat,
        dst_feat,
        voxel_size,
        fast,
        icp_only=icp_only,
        init_pose=init_pose,
        throw_if_icp=throw_if_icp,
        icp_thresh=icp_thresh,
    )
    result = np.linalg.inv(result)
    if get_info:
        return test(result, poses[-1]), result, info
    else:
        return test(result, poses[-1]), result


def solve_fpfh_with_given_pcds(
    pcd0,
    pcd1,
    voxel_size: float = 0.05,
    get_info: bool = False,
    fast: bool = False,
    **kwargs,
):
    pcd0, f0 = pcd0
    pcd1, f1 = pcd1
    result, info, fit = pairwise_registration(
        pcd0, pcd1, f0, f1, voxel_size, fast, ret_result=True, **kwargs
    )
    result = np.linalg.inv(result)
    if get_info:
        return result, info, fit
    return result
