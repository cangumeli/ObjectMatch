import os
from functools import lru_cache
from typing import Optional

import cv2 as cv
import numpy as np
import open3d as o3d
import quaternion
import torch
from PIL import Image


def rotate_x(degree):
    angle = np.deg2rad(degree)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])


def rotate_y(degree):
    angle = np.deg2rad(degree)
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])


def rotate_z(degree):
    angle = np.deg2rad(degree)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def back_project(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = np.linspace(0, depth.shape[1] - 1, depth.shape[1])
    y = np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    x, y = np.meshgrid(x, y)
    z = depth
    xyz = np.stack([x * z, y * z, z], -1)
    return xyz @ np.linalg.inv(K).T


def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M


def decompose_mat4(M: np.ndarray, ret_q=False) -> tuple:
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    if ret_q:
        q = quaternion.as_float_array(quaternion.from_rotation_matrix(R[0:3, 0:3]))
        # q = quaternion.from_float_array(quaternion_from_matrix(M, False))
    else:
        q = R[0:3, 0:3]

    t = M[0:3, 3]
    return t, q, s


def load_matrix(fname: str) -> np.ndarray:
    with open(fname) as f:
        mat = [[float(v) for v in line.strip().split()] for line in f]
    return np.asarray(mat, dtype=np.float32)


def load_16bit(fname: str) -> np.ndarray:
    img = cv.imread(fname, -1)
    if img is None:
        raise FileNotFoundError(fname + ' is not an image')
    return img


def load_1channel(fname: str) -> np.ndarray:
    img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(fname + ' is not an image')
    return img


def remove_outliers(pts: np.ndarray, neigbors: int = 10, std_ratio: float = 2.):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    new_pcd, idx = pcd.remove_statistical_outlier(neigbors, std_ratio)
    mask = np.zeros(pts.shape[0], dtype=np.bool_)
    mask[idx] = True
    return np.asarray(new_pcd.points, dtype=pts.dtype), mask


def save_nocs(filename: str, nocs: np.ndarray, mask: Optional[np.ndarray] = None):
    if mask is None:
        mask = nocs != 0
    elif mask.ndim == 2:
        mask = mask[..., None]
    nocs = mask * (nocs + 0.5) * 10000
    nocs = nocs.round().astype(np.uint16)
    cv.imwrite(filename, nocs)


def load_nocs(filename: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
    nocs = cv.imread(filename, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    if nocs is None:
        raise FileNotFoundError(filename + ' is not an image')
    
    if mask is None:
        mask = np.any(nocs != 0, -1, keepdims=True)
    elif mask.ndim == 2:
        mask = mask[..., None]
    nocs = mask * (nocs / 10000 - 0.5)
    return nocs


def load_depth(filename: str, dtype=np.float64, div: float = 1000.) -> np.ndarray:
    if not os.path.isfile(filename):
        raise FileNotFoundError
    depth = load_16bit(filename)
    depth = depth.astype(dtype)
    return depth / div


@lru_cache(maxsize=500)
def load_depth_cached(filename: str) -> np.ndarray:
    if not os.path.isfile(filename):
        raise FileNotFoundError
    depth = load_16bit(filename)
    depth = depth.astype('float32')
    return depth / 1000


def image_grid(imgs: list, rows: int, cols: int):
    # assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def median_filter(signal: torch.Tensor, thresh=3) -> torch.Tensor:
    median = signal.median()
    diff = torch.abs(signal - median)
    s = diff / diff.median().clamp(1e-5)
    return (s < thresh) | (signal < median)


def load_pose_files(data_dir: str, scene: str, trajectory: list[int]) -> list[np.ndarray]:
    pose_files = [os.path.join(data_dir, scene, 'pose', '{}.txt'.format(i)) for i in trajectory]
    poses = [load_matrix(f) for f in pose_files]
    to_1st_camera = np.linalg.inv(poses[0])
    poses = [to_1st_camera @ pose for pose in poses]
    return poses


def draw_nocs(img0, img1, res0, res1, rat=.8, vertical=False, pad_size=0, ret_img=False):
    stack_fn = np.vstack if vertical else np.hstack
    if vertical:
        pad = 255 * np.ones((pad_size, img0.shape[1], 3), dtype='uint8')
    else:
        pad = 255 * np.ones((img0.shape[0], pad_size, 3), dtype='uint8')
    if len(res0):
        img0_, img1_ = img0.astype(np.float32).copy(), img1.astype(np.float32).copy()
        # exit()
        for res, img in zip((res0, res1), (img0_, img1_)):
            img_ = np.zeros_like(img)
            for xy, noc, box, mask in zip(
                res.pred_xy_grids.tensor.cpu().numpy(),
                res.pred_nocs.tensor.cpu().numpy(),
                res.pred_boxes.tensor.cpu().numpy(),
                res.pred_masks.cpu().numpy(),
            ):
                noc = 255 * (.5 + noc.reshape(3, -1).T)  # [..., [1, 0, 2]]  # [..., ::-1]
                xy = xy.reshape(2, -1).T.astype(np.int64).tolist()
                x0, y0, x1, y1 = box.tolist()
                offset_x = max(int((x1 - x0) / 32), 1)
                offset_y = max(int((y1 - y0) / 32), 1)
                for (x, y), n in zip(xy, noc):
                    if np.allclose(n, 0):
                        continue
                    img_[y-offset_y:y+offset_y+1, x-offset_x:x+offset_x+1] = n
                # np.copyto(img, cv.bilateralFilter(img, 9, 75, 75))
            
                # img_ = cv.GaussianBlur(img_, (11, 11), -1)
                img_ = cv.medianBlur(img_.astype('uint8'), 23)
                # img_ = cv.bilateralFilter(img_, 9, 75, 75)
                mask = (img_ != 0).any(-1) & mask
                img[mask] = (1 - rat) * img[mask] + rat * img_[mask]
        '''for mask0, mask1 in zip(
            res0.pred_masks.cpu().numpy(),
            res1.pred_masks.cpu().numpy(),
        ):
            img0_[mask0] = 0.5 * img0_[mask0] + np.array([100, 0, 0])
            img1_[mask1] = 0.5 * img1_[mask1] + np.array([100, 0, 0])'''
        img_ = stack_fn([img0_.astype('uint8'), pad, img1_.astype('uint8')])
    else:
        img_ = stack_fn([img0, pad, img1])
    if ret_img:
        raw_img = stack_fn([img0, pad, img1])
        return img_, raw_img
    return img_
