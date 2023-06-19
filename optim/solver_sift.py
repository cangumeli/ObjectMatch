import os
from typing import Any

import cv2 as cv
import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix

from optim.common import load_depth, load_matrix
from optim.solver import distance_filter, overlap_energy, refine_icp, test
from optim.solver_fpfh import load_point_clouds


def extract_sift(images: list[np.ndarray], num_levels=30, orb=False, **kwargs):
    if orb:
        sift = cv.ORB_create(nlevels=30)
    else:
        sift = cv.SIFT_create(nOctaveLayers=num_levels, **kwargs)
    grays = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    return [sift.detectAndCompute(gray, None) for gray in grays]


def match_features(
    sift_features: list[tuple[Any, Any]],
    depth_images: list[np.ndarray],
    intr: np.ndarray,
    thresh: float = 1.,
    min_points: int = 5,
    ratio_test: bool = False,  # True,
    ratio_thresh: float = .75,
):
    matches = []
    matcher = cv.BFMatcher()
    intr_inv = np.linalg.inv(intr)[:3, :3]
    for i in range(len(sift_features) - 1):
        for j in range(i + 1, len(sift_features)):
            # if j != i + 1:
            #    continue
            kpi, di = sift_features[i]
            kpj, dj = sift_features[j]
            mij = matcher.knnMatch(di, dj, k=2)
            good = []
            src_points = []
            dst_points = []
            for m, n in mij:
                if ratio_test and not (m.distance < ratio_thresh * n.distance):
                    continue
                else:
                    good.append([m])
                src = kpi[m.queryIdx]
                dst = kpj[m.trainIdx]
                x1, y1 = map(round, src.pt)
                x2, y2 = map(round, dst.pt)
                z1 = depth_images[i][int(y1), int(x1)]
                z2 = depth_images[j][int(y2), int(x2)]
                if z1 < 1e-5 or z2 < 1e-5:
                    good.pop()
                    continue
                src_points.append(intr_inv @ np.array([x1, y1, z1]))
                dst_points.append(intr_inv @ np.array([x2, y2, z2]))


            mask = np.ones(len(good), dtype=np.bool_)
            src_points = np.vstack(src_points)
            dst_points = np.vstack(dst_points)

            # import pdb; pdb.set_trace()

            new_mask, err = distance_filter(
                src_points,
                dst_points,
                mask,
                thresh=thresh,
            )

            if new_mask.sum() < min_points:
                continue
            good = [g for g, b in zip(good, new_mask.tolist()) if b]
            matches.append((
                i,
                j,
                src_points[new_mask].astype(np.float32),
                dst_points[new_mask].astype(np.float32),
                good,
                # [g[0] for g in good],
                err.float(),
            ))

    return matches


def create_overlaps(matches):
    overlaps = []
    for i, j, di, dj, m_i, err in matches:
        # weights = torch.as_tensor([1 / m.distance for m in m_i])
        weights = torch.as_tensor([1. - e for e in err])
        weights = torch.ones_like(weights)
        # weights /= torch.sum(weights)
        # weights *= torch.as_tensor([1 / e for e in err.tolist()])
        weights /= torch.norm(weights)
        overlaps.append((i, j, torch.from_numpy(di), torch.from_numpy(dj), weights))
    return overlaps


def solve_sift_pair(
    data_dir=None,
    scene=None,
    i=None,
    j=None,
    num_iters=50,
    images=None,
    depth_images=None,
    feat=None,
    intr=None,
    pcd=None,
    gt_pose=None,
    thresh=.3,
    min_points=3,
    throw_if_icp=False,
):

    if images is None:
        images = [
            cv.imread(os.path.join(data_dir, scene, 'color', '{}.jpg'.format(i))),
            cv.imread(os.path.join(data_dir, scene, 'color', '{}.jpg'.format(j)))
        ]
    
    if depth_images is None:
        depth_images = [
            load_depth(os.path.join(data_dir, scene, 'depth', '{}.png'.format(i))),
            load_depth(os.path.join(data_dir, scene, 'depth', '{}.png'.format(j))),
        ]

    if feat is None:
        feat = extract_sift(images)

    if intr is None:
        try:
            intr = load_matrix(os.path.join(data_dir, scene, 'intrinsic_depth.txt'))
        except FileNotFoundError:
            intr = load_matrix(os.path.join(data_dir, scene, 'intrinsic/intrinsic_depth.txt'))

    try:
        matches = match_features(feat, depth_images, intr, thresh=thresh, min_points=min_points)
    except:
        matches = []

    if not len(matches):
        if throw_if_icp:
            raise RuntimeError('ICP')
        print('Falling back to ICP')
        voxel_size = 0.05
        if pcd is not None:
            assert gt_pose is not None
            src_down, dst_down = pcd
        else:
            [(src_down, _), (dst_down, _)], poses = load_point_clouds(
                data_dir, scene, [i, j], feature=False, voxel_size=voxel_size
            )
            gt_pose = poses[-1]
        result = refine_icp(
            np.eye(4),
            src_down,
            dst_down,
            voxel_size=voxel_size,
            preprocess=False,
            thresh=.1,
        )
        '''result = refine_icp(
            result,
            src_down,
            dst_down,
            voxel_size=voxel_size,
            preprocess=False,
            thresh=.01
        )'''
        return test(result, gt_pose), result

    print(len(matches[0][2]))

    overlaps = create_overlaps(matches)
    loss_history = []
    dumping_factor = 1e-5
    cam_angles = torch.zeros(1, 3)
    cam_transes = torch.zeros_like(cam_angles)
    for k in range(num_iters):
        jacob, res = overlap_energy(overlaps, cam_angles, cam_transes)[:2]
        if k > 15:
            res_ = res.reshape(-1, 3).clone()
            flt = res_.norm(dim=-1) < .15
            jacob = jacob[flt.repeat_interleave(3)]
            res = res[flt.repeat_interleave(3)]
            # print('Keep', flt.float().mean())'''
        try:
            jtj = jacob.t() @ jacob
            jtf = jacob.t() @ res
            if dumping_factor > 0:
                jtj = jtj + dumping_factor * torch.eye(jtj.size(0), device=jtj.device) * jtj
            step = torch.solve(jtf, 2 * jtj).solution.flatten()
        except RuntimeError as e:
            print(e, '\n@ iter {}'.format(k))
            break

        num_views = 2
        cam_angles = cam_angles - torch.cat([s[:3] for s in step.chunk(num_views - 1)])\
            .view_as(cam_angles).cpu()
        cam_transes = cam_transes - torch.cat([s[3:6] for s in step.chunk(num_views - 1)])\
            .view_as(cam_transes).cpu()

        loss_history.append(res.square().sum().item())
        if len(loss_history) >= 2:
            dumping_factor *= (0.1 if loss_history[-1] < loss_history[-2] else 10)

    mats = euler_angles_to_matrix(cam_angles, 'XYZ').squeeze().numpy()
    trs = cam_transes.squeeze().numpy()
    pose = np.eye(4)
    pose[:3, :3] = mats
    pose[:3, 3] = trs

    voxel_size = 0.05
    thresh = 0.1
    if pcd is None:
        [(src_down, _), (dst_down, _)], poses = load_point_clouds(
            data_dir, scene, [i, j], feature=False, voxel_size=voxel_size
        )
        gt_pose = poses[-1]
        
    else:
        assert gt_pose is not None
        src_down, dst_down = pcd
    result = refine_icp(
        pose, src_down, dst_down, voxel_size=voxel_size, preprocess=False, thresh=thresh
    )    

    return test(result, gt_pose), result
