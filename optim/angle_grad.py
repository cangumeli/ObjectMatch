import torch
from numpy import deg2rad
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    random_rotations,
)


def loss(R, c, P, D, reg=0.1):
    diffs = torch.sum((P @ R.t() - D) ** 2, -1)
    return 0.5 * (torch.dot(c, diffs) + reg * torch.sum((1 - c)**2))


def rotation_grad(R, c, P, D, reg):
    ''''
    Args:
        R: 3x3
        c: N
        P: Nx3
        D: Nx3
    '''
    diffs = P @ R.t() - D
    dR = (c[..., None] * diffs).t() @ P
    dc = 0.5 * (diffs ** 2).sum(-1) + reg * (c - 1)
    return dR, dc


def make_rotation_mat(angles: torch.Tensor):
    rx, ry, rz = angles.chunk(3, -1)

    cosx = torch.cos(rx)
    sinx = torch.sin(rx)

    cosy = torch.cos(ry)
    siny = torch.sin(ry)

    cosz = torch.cos(rz)
    sinz = torch.sin(rz)

    matrix = [
        cosy*cosz,                   -cosy*sinz,                  siny,
        cosx*sinz + cosz*sinx*siny, cosx*cosz - sinx*siny*sinz, -cosy*sinx,
        -cosx*cosz*siny + sinx*sinz, cosx*siny*sinz + cosz*sinx,  cosx*cosy
    ]

    matrix = torch.cat(matrix, -1).reshape(*angles.shape, 3)  # .reshape(-1, 3, 3)
    if matrix.size(0) == 1:
        matrix = matrix.reshape(3, 3)
    return matrix, (cosx, sinx, cosy, siny, cosz, sinz)


def angle_grad(dmat, cosx, sinx, cosy, siny, cosz, sinz):

    def d(i, j):
        return dmat[..., i, j]

    dcosx = (
          d(1, 0) * sinz
        + d(1, 1) * cosz
        - d(2, 0) * cosz * siny
        + d(2, 1) * siny * sinz
        + d(2, 2) * cosy
    )
    dsinx = (
          d(1, 0) * cosz * siny
        - d(1, 1) * siny * sinz
        - d(1, 2) * cosy 
        + d(2, 0) * sinz
        + d(2, 1) * cosz
    )

    dcosy = (
          d(0, 0) * cosz
        - d(0, 1) * sinz
        - d(1, 2) * sinx
        + d(2, 2) * cosx
    )
    dsiny = (
          d(0, 2)
        + d(1, 0) * cosz * sinx
        - d(1, 1) * sinx * sinz
        - d(2, 0) * cosx * cosz
        + d(2, 1) * cosx * sinz
    )

    dcosz = (
          d(0, 0) * cosy
        + d(1, 0) * sinx * siny
        + d(1, 1) * cosx
        - d(2, 0) * cosx * siny
        + d(2, 1) * sinx
    )
    dsinz = (
        - d(0, 1) * cosy
        + d(1, 0) * cosx
        - d(1, 1) * sinx * siny
        + d(2, 0) * sinx
        + d(2, 1) * cosx * siny
    )

    drx = dsinx * cosx - dcosx * sinx
    dry = dsiny * cosy - dcosy * siny
    drz = dsinz * cosz - dcosz * sinz

    return torch.stack((drx, dry, drz), dim=-1)


# torch.manual_seed(20)

if __name__ == '__main__':
    nocs = torch.rand(10, 3)
    mat = random_rotations(1).squeeze_()
    depth = nocs @ mat.t()
    eye_mat = torch.eye(3)
    weights = torch.rand(10)
    reg = 0.1

    eye_mat.requires_grad_()
    weights.requires_grad_()
    cost = loss(eye_mat, weights, nocs, depth, reg)
    cost.backward()

    print(eye_mat.grad)
    rot_grad, weight_grad = rotation_grad(eye_mat, weights, nocs, depth, reg)
    print(rot_grad)

    print()
    print(weights.grad)
    print(weight_grad)

    print()
    print(matrix_to_euler_angles(mat, 'XYZ'))
    # print(loss(eye_mat, weights, nocs, depth))

    angles = torch.tensor([[deg2rad(30), deg2rad(15), deg2rad(40)]]).float().requires_grad_()

    pt3d = euler_angles_to_matrix(angles, 'XYZ').squeeze_()
    loss(pt3d, weights, nocs, depth).backward()

    mine, cache = make_rotation_mat(angles)
    dmat, dweights = rotation_grad(mine, weights, nocs, depth, reg)
    dangles = angle_grad(dmat, *cache)

    print()
    print(angles.grad.squeeze())
    print(dangles)
    # import IPython; IPython.embed()
    # print(torch.isclose(mine, pt3d.reshape(3, 3)))
