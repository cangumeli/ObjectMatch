from typing import Optional

import torch

from optim.angle_grad import angle_grad, make_rotation_mat


def residuals(R, c, P, D, t=None) -> torch.Tensor:
    diffs = P @ R.transpose(-2, -1) - D
    if t is not None:
        diffs = diffs + t
    norms = torch.norm(diffs, dim=-1)
    norms = c.clamp(1e-5).sqrt() * norms
    return norms


def residual_losses(R, c, P, D, t=None, ret_res=False, reg=0):
    res = residuals(R, c, P, D, t)
    result = ({
        'data': torch.sum(res ** 2),
        'reg': reg * torch.sum((1 - c)**2) if reg > 0 else 0.0
    }, res)
    if not ret_res:
        result = result[0]
    return result


def standard_losses(R, c, P, D, t=None, reg=0) -> dict[str, torch.Tensor]:
    # data_loss = 
    if t is None:
        t = torch.zeros(3)
    diffs = (P @ R.transpose(-2, -1) + t - D)**2
    return {
        'data': torch.sum(c * diffs.sum(-1)),
        'reg': reg * torch.sum((1 - c)**2)
    }


def jacobian(res, R, R_cache, c, P, D):
    # First write a non-batched version
    jacob = []
    for ri, pi, di, ci in zip(res, P, D, c):
        diff = R @ pi - di
        # li = torch.sum(diff**2)
        ei = ri / ci.sqrt()  # li.sqrt().clamp(1e-5)

        dei = ci.sqrt()
        dli = dei / (2 * ei)
        dR = dli * 2 * (diff[:, None] @ pi[None, :])
        da = angle_grad(dR, *R_cache)
        jacob.append(da)

    return torch.stack(jacob)


def jacobian_vec(
    res, R, R_cache, c, P, D, t=None, s=None
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    diffs = P @ R.transpose(-2, -1) - D
    if t is not None:
        diffs = diffs + t

    dL = c / res.clamp(1e-5)
    dR = dL[..., None, None] * (diffs[..., None] @ P[..., None, :])
    da = angle_grad(dR, *R_cache)

    if t is not None:
        dt = dL[..., None] * diffs
        return da, dt

    return da, None


def jacobian_vec_9dof(
    res: torch.Tensor,
    t: torch.Tensor,
    R: torch.Tensor,
    R_cache,
    s: torch.Tensor,
    c: torch.Tensor,
    P: torch.Tensor,
    D: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # s_exp = s.exp()
    P = P * s.mul(1e-1).exp()
    diffs = P @ R.transpose(-2, -1) - D
    diffs = diffs + t

    dL = c / res.clamp(1e-5)
    dR = dL[..., None, None] * (diffs[..., None] @ P[..., None, :])
    da = angle_grad(dR, *R_cache)

    dt = dL[..., None] * diffs

    ds = dL[..., None] * diffs * P * 1e-1

    return dt, da, ds


def residuals2(P1: torch.Tensor, P2: torch.Tensor, c: torch.Tensor):
    if c.ndim == P1.ndim - 1:
        c = c[..., None]
    return c * (P1 - P2)


def jacobian2(
    c: torch.Tensor,
    R: torch.Tensor,
    R_cache,
    t: torch.Tensor,
    P: torch.Tensor,
    D: torch.Tensor,
    res: Optional[torch.Tensor] = None,
):
    if res is None:
        res = c * (P @ R.squeeze().transpose(-2, -1) + t - D)
    else:
        res = res.reshape_as(P)

    dR = torch.zeros(*res.size()[:-1], 3, 9, device=res.device, dtype=res.dtype)
    cP = c * P
    dR[..., 0, :3] = cP
    dR[..., 1, 3:6] = cP
    dR[..., 2, 6:9] = cP
    if res.ndim < dR.ndim-1:
        dR = dR.squeeze()

    dR = dR.reshape(*res.size()[:-1], 3, 3, 3)
    da = angle_grad(dR, *R_cache)

    dt = torch.eye(3, device=res.device, dtype=res.dtype)
    dt = dt * c.unsqueeze(-1)

    return res.reshape(-1, 1), da.reshape(-1, 3), dt.reshape(-1, 3)


def jacobian2_9dof(
    c: torch.Tensor,
    R: torch.Tensor,
    R_cache,
    t: torch.Tensor,
    s: torch.Tensor,
    P: torch.Tensor,
    D: torch.Tensor,
    res: Optional[torch.Tensor] = None,
):
    ssign = s.sign()
    s = s.abs()

    if res is None:
        res = c * ((s * P) @ R.squeeze().transpose(-2, -1) + t - D)
    else:
        res = res.reshape_as(P)

    # Cached variables
    cP = c * P
    cPs = cP * s

    # Rotation jacobian
    dR = torch.zeros(*res.size()[:-1], 3, 9, device=res.device, dtype=res.dtype)
    dR[..., 0, 0:3] = cPs
    dR[..., 1, 3:6] = cPs
    dR[..., 2, 6:9] = cPs
    dR = dR.reshape(-1, 3, 3)
    if res.ndim == 2:  # No batching
        dR = dR.squeeze(0)
    da = angle_grad(dR, *R_cache)

    # Translation jacobian
    dt = c.unsqueeze(-1) * torch.eye(3, device=res.device, dtype=res.dtype)

    # Scale jacobian
    ds = cP.unsqueeze(-2) * R
    ds = ds.reshape(-1, 3) * ssign

    return res.reshape(-1, 1), da.reshape(-1, 3), dt.reshape(-1, 3), ds.reshape(-1, 3)


def jacobian2_9dof_vec(
    c: torch.Tensor,
    R: torch.Tensor,
    R_cache,
    t: torch.Tensor,
    s: torch.Tensor,
    P: torch.Tensor,
    D: torch.Tensor,
    res: Optional[torch.Tensor] = None,
):
    ssign = s.sign()
    s = s.abs()

    pad_size = P.shape[:-1]
    if res is None:
        res = c * ((s * P) @ R.squeeze().transpose(-2, -1) + t - D)
    else:
        res = res.reshape_as(P)

    # Cached variables
    cP = c * P
    cPs = cP * s

    # Translation jacobian
    dt = c.unsqueeze(-1) * torch.eye(3, device=res.device, dtype=res.dtype)

    # Rotation jacobian
    dR = torch.zeros(*pad_size, 3, 9, device=res.device, dtype=res.dtype)
    dR[..., 0, 0:3] = cPs
    dR[..., 1, 3:6] = cPs
    dR[..., 2, 6:9] = cPs
    dR = dR.reshape(*dR.shape[:-1], 3, 3)
    da = angle_grad(dR, *R_cache)

    # Scale jacobian
    ds = cP.unsqueeze(-2) * R * ssign.unsqueeze(-2)

    return res, da, dt, ds
