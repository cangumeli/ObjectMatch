import os
from collections import namedtuple
from typing import Optional
import warnings

import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.transforms import matrix_to_euler_angles, so3_relative_angle

from optim.angle_grad import make_rotation_mat
from optim.constants import GRID_SIZE
from optim.gauss_newton import jacobian2, jacobian2_9dof_vec


InstanceDatum = namedtuple(
    'InstanceDatum', ['nocs', 'depths', 'masks', 'ids', 'classes']
)
InstanceData = list[InstanceDatum]
_BatchedInstanceData = tuple[
    list[int], torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]
]
DEVICE = 'cuda' if torch.has_cuda else 'cpu'


def batch_instance_data(instance_data: InstanceData) -> _BatchedInstanceData:
    c_list = []
    all_nocs = []
    all_depths = []
    all_ids = []
    all_cams = []
    num_per_frame = []

    for i, (nocs, depths, masks, ids, _) in enumerate(instance_data):
        c = masks.flatten(1).float()  # torch.ones(nocs.size(0), nocs.size(1), 1, device=DEVICE)
        c.div_(c.norm())
        c_list.append(c)
        all_nocs.append(nocs.flatten(2).transpose(1, 2))
        all_depths.append(depths.flatten(2).transpose(1, 2))
        all_ids.extend(ids)
        if i > 0:
            all_cams.extend(i-1 for _ in ids)
        num_per_frame.append(len(ids))

    # Batched jacobians
    nocs = torch.cat(all_nocs)
    depths = torch.cat(all_depths)
    c = torch.cat(c_list)
    # c = len(instance_data) * c / c.norm()
    return all_ids, nocs, depths, c, all_cams, num_per_frame


def frame_energy(
    batched_instance_data: _BatchedInstanceData,  #: InstanceData,
    cam_angles: torch.Tensor,
    cam_transes: torch.Tensor,
    obj_angles: torch.Tensor,
    obj_transes: torch.Tensor,
    obj_scales: torch.Tensor,
    obj_dof: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:

    all_ids, nocs, depths, c, all_cams, num_per_frame = batched_instance_data
    # batch_instance_data(instance_data)

    ids = all_ids
    # print(ids)
    num_first = num_per_frame[0]
    zeros = torch.zeros(num_first, 3, device=DEVICE)
    cam_rot = torch.cat([zeros, cam_angles[all_cams]]).unsqueeze(1)
    cam_rot, cam_cache = make_rotation_mat(cam_rot)
    cam_trs = torch.cat([zeros, cam_transes[all_cams]]).unsqueeze(1)

    try:
        obj_rot = obj_angles[[id-1 for id in ids]].unsqueeze(1)
    except RuntimeError:
        from IPython import embed; embed()
    obj_rot, obj_cache = make_rotation_mat(obj_rot)
    obj_trs = obj_transes[[id-1 for id in ids]].unsqueeze(1)
    obj_scale = obj_scales[[id-1 for id in ids]].unsqueeze(1)

    depths_ = depths @ cam_rot.squeeze(1).transpose(-1, -2) + cam_trs
    nocs_ = (nocs * obj_scale) @ obj_rot.squeeze(1).transpose(-1, -2) + obj_trs

    _, da, dt = jacobian2(
        c.unsqueeze(-1),
        cam_rot,
        cam_cache,
        cam_trs,
        depths,
        nocs_,
    )
    da, dt = -da, -dt
    jacob_cam = torch.zeros(
        (np.prod(GRID_SIZE) * 3 * len(all_ids), 6 * len(num_per_frame) - 6),
        device=DEVICE
    )
    start = 0
    for i, num in enumerate(num_per_frame):
        end = start + num * np.prod(GRID_SIZE) * 3
        if i > 0:
            cstart = 6 * (i - 1)
            jacob_cam[start:end, cstart:cstart+3] = da[start:end]
            jacob_cam[start:end, cstart+3:cstart+6] = dt[start:end]
        start = end

    # if obj_dof == 9:
    res, da, dt, ds = jacobian2_9dof_vec(
        c.unsqueeze(-1),
        obj_rot,
        obj_cache,
        obj_trs,
        obj_scale,
        nocs,
        depths_,
    )
    '''elif obj_dof == 6:
        res, da, dt = jacobian2(
            c.unsqueeze(-1),
            obj_rot,
            obj_cache,
            obj_trs,
            nocs_,
            depths,
        )
    else:
        print('{} DoF is not supported'.format(obj_dof))'''

    # jacob_cam = torch.cat(jacob_list)
    jacob_object = torch.zeros(
        (np.prod(GRID_SIZE) * 3 * len(all_ids), obj_dof * len(set(all_ids))),
        device=DEVICE
    )
    start = 0
    for k, id in enumerate(all_ids):
        end = start + np.prod(GRID_SIZE) * 3
        cstart = obj_dof * (id - 1)
        # import pdb; pdb.set_trace()
        jacob_object[start:end, cstart:cstart+3] = da[k].reshape(-1, 3)
        jacob_object[start:end, cstart+3:cstart+6] = dt[k].reshape(-1, 3)
        if obj_dof == 9:
            jacob_object[start:end, cstart+6:cstart+9] = ds[k].reshape(-1, 3)
        start = end

    jacob = torch.cat([jacob_cam, jacob_object], -1)

    return jacob, res.reshape(jacob.size(0), -1)


def preprocess_pcd(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def proc_init(
    nocs: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    scales: torch.Tensor,
    ret_error: bool = False,
    ret_angles = True,
):
    nocs = nocs * scales.reshape(-1, 3, 1, 1)
    nocs_pts = nocs.flatten(2).transpose(1, 2)
    depth_pts = depths.flatten(2).transpose(1, 2)
    weights = masks.flatten(1)

    trs = corresponding_points_alignment(
        nocs_pts.double(),
        depth_pts.double(),
        weights.double(),
    )
    t, r = trs.T.float(), trs.R.float()

    if ret_error:
        errs: torch.Tensor = torch.norm(nocs_pts @ r + t - depth_pts, dim=-1) * weights
        if ret_angles:
            r = matrix_to_euler_angles(r, 'XYZ')
        return t, r, errs
    else:
        if ret_angles:
            r = matrix_to_euler_angles(r, 'XYZ')
        return t, r


def filter_and_initalize(
    pred_nocs: torch.Tensor,
    pred_depths: torch.Tensor,
    masks: torch.Tensor,
    pred_scales: torch.Tensor,
    filter_coef: float = 0.2,
    min_mask: int = 30,
    ret_angles: bool = False,
):
    # TODO: Parallelize
    transes, angles = [], []
    new_masks = []
    converged = []
    for noc, depth, mask, scale in zip(
        pred_nocs.unbind(), pred_depths.unbind(), masks.unbind(), pred_scales.unbind()
    ):
        mask_ = mask
        mask_size = mask.gt(0).sum().item()
        while mask_size >= min_mask:
            # import pdb; pdb.set_trace()
            t, a, errs = proc_init(
                noc[None],
                depth[None],
                mask[None],
                scale[None],
                ret_angles=ret_angles,
                ret_error=True,
            )
            errs = errs.view_as(mask)
            err_mask = errs <= filter_coef
            if err_mask.all():
                transes.append(t.squeeze())
                angles.append(a.squeeze())
                converged.append(True)
                new_masks.append(mask)
                break
            mask = err_mask * mask
            mask_size = mask.gt(0).sum().item()
        if mask_size < min_mask:
            transes.append(torch.zeros(3, device=noc.device))
            if ret_angles:
                angles.append(torch.zeros(3, device=noc.device))
            else:
                angles.append(torch.eye(3, device=noc.device))
            converged.append(False)
            new_masks.append(mask_)

    return (
        torch.stack(transes),
        torch.stack(angles),
        converged,
        torch.cat(new_masks),
    )


def refine_icp(
    init_transform: np.ndarray,
    depth_pcd0: np.ndarray,
    depth_pcd1: np.ndarray,
    voxel_size: float = 0.05,
    thresh: float = 0.1,
    get_info: bool = False,
    preprocess: bool = True,
) -> np.ndarray:
    if preprocess:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(depth_pcd0)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(depth_pcd1)

        pcd0 = preprocess_pcd(pcd0, voxel_size)
        pcd1 = preprocess_pcd(pcd1, voxel_size)
    else:
        pcd0, pcd1 = depth_pcd0, depth_pcd1

    result = o3d.pipelines.registration.registration_icp(
        pcd1, pcd0, thresh,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    if get_info:
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcd1, pcd0, thresh,
            result.transformation,
        )
        return result.transformation, info

    return result.transformation


def test(pred_pose: np.ndarray, gt_pose: np.ndarray) -> tuple[float, float]:
    try:
        angle_diff = torch.rad2deg(so3_relative_angle(
            torch.from_numpy(pred_pose[:3, :3][None]).double(),
            torch.from_numpy(gt_pose[:3, :3][None]).double(),
        )).item()
    except ValueError:
        angle_diff = 0.
    trans_diff = 100 * np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
    return trans_diff, angle_diff


Overlaps = list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]


def distance_filter(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    mask: np.ndarray,
    thresh: float = 0.05,
    ret_trs: bool = False,
    max_iter = None,
):
    mask = torch.from_numpy(mask).clone()
    src_pts, dst_pts = map(torch.from_numpy, (src_pts, dst_pts))
    sim = None
    err = torch.ones_like(mask).float()
    step = 0
    while mask.any():
        if max_iter is not None and step >= max_iter:
            break
        step += 1
        dtype = src_pts.dtype
        with warnings.catch_warnings(record=True) as w:
            sim = corresponding_points_alignment(
                src_pts[None],
                dst_pts[None],
                mask[None].to(dtype=dtype)
            )
        if len(w):
            msg = str(w[-1].message).lower()
            if "cannot return a unique rotation" in msg or "low rank" in msg:
                warnings.warn('Failed with: {}'.format(msg))
                if ret_trs:
                    return torch.zeros_like(mask).numpy(), None, np.eye(4)
                else:
                    return torch.zeros_like(mask).numpy(), None

        err = torch.norm(src_pts @ sim.R.squeeze() + sim.T.squeeze() - dst_pts, dim=-1)
        remove = err > thresh
        if not torch.any(mask & remove):
            break
        mask.logical_and_(~remove)

    rel_pose = np.eye(4)
    if sim is not None:
        rel_pose[:3, :3] = sim.R.squeeze().t().numpy()
        rel_pose[:3, 3] = sim.T.squeeze().numpy()

    if ret_trs:
        return mask.numpy(), err[mask], rel_pose

    return mask.numpy(), err[mask]


def overlap_energy(
    overlaps: Overlaps,
    cam_angles: torch.Tensor,
    cam_transes: torch.Tensor,
    enable_filter: bool = False,
    outlier_thresh: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, set]:

    jacob_list = []
    res_list = []

    removed = set()

    device = cam_angles.device

    for i, j, di, dj, w in overlaps:
        # w = torch.ones_like(w)
        i = i - 1
        j = j - 1
        if i >= 0:
            mat_i, cache_i = make_rotation_mat(cam_angles[i])
            trs_i = cam_transes[i]
            di_ = di @ mat_i.t() + trs_i
        else:
            di_ = di
        mat_j, cache_j = make_rotation_mat(cam_angles[j])
        trs_j = cam_transes[j]
        dj_ = dj @ mat_j.t() + trs_j
        w = w.reshape(-1, 1)
        # res_list.append(w.mul(dj_ - di_).flatten())

        # jacob = torch.zeros(3 * w.numel() , 6 * cam_angles.size(0))
        jacob = torch.zeros(3 * w.numel(), 6 * cam_angles.size(0), device=device)
        if i >= 0:
            res, da_i, dt_i = jacobian2(
                w,
                mat_i,
                cache_i,
                trs_i,
                di,
                dj_,
            )
            # if enable_filter and res.abs().max() > outlier_thresh:
            # import pdb; pdb.set_trace()
            # if enable_filter and torch.any(res.abs() > w.repeat_interleave(3) * outlier_thresh):
            if enable_filter and torch.any(
                res.reshape(3, -1).norm(dim=0) > w.flatten() * outlier_thresh
            ):
                removed.add((i + 1, j + 1))
                continue

            start = i * 6
            jacob[:, start:start + 3] = -da_i
            jacob[:, start + 3:start + 6] = -dt_i

        # import pdb; pdb.set_trace()
        res, da_j, dt_j = jacobian2(
            w,
            mat_j,
            cache_j,
            trs_j,
            dj,
            di_,
        )

        # if enable_filter and res.abs().max() > outlier_thresh:
        # if enable_filter and torch.any(res.abs() > w.repeat_interleave(3) * outlier_thresh):
        if enable_filter and torch.any(res.reshape(3, -1).norm(dim=0) > w.flatten() * outlier_thresh):
            removed.add((i + 1, j + 1))
            continue

        res_list.append(res)
        start = j * 6
        jacob[:, start:start + 3] = da_j
        jacob[:, start + 3:start + 6] = dt_j
        jacob_list.append(jacob)

    return torch.cat(jacob_list), torch.cat(res_list), removed


def pose_accuracy(errors: list[float], thresholds: list[int]):
    result: list[float] = []
    for t in thresholds:
        result.append(np.mean([e <= t for e in errors]))
    return result


def pose_recall(
    errors_pairs: list[tuple[float, float]],
    t_thresh: list[float],
    a_thresh: list[float],
):
    result: list[float] = []
    for t, a in zip(t_thresh, a_thresh):
        result.append(np.mean([te <= t and ae <= a for te, ae in errors_pairs]))
    return result
