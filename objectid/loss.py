from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def build_loss(cfg: dict) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    loss_cfg = cfg['SOLVER']['LOSS']
    margin = loss_cfg['MARGIN']
    name_: str = loss_cfg['NAME']
    name = name_.lower()

    print('Building loss: {}...'.format(name_))

    def loss(a: Tensor, p: Tensor, n: Tensor) -> Tensor:
        if name == 'triplet':
            return F.triplet_margin_loss(a, p, n, margin)

        elif name == 'triplet_focal':
            pow = loss_cfg['FOCAL_POW']
            diffs = F.triplet_margin_loss(a, p, n, margin, reduction='none')
            return diffs.div(margin).pow(pow).mean()

        elif name == 'embedding':
            diffs: Tensor = torch.norm(torch.cat([a, a]) - torch.cat([p, n]), dim=-1)
            labels = torch.ones(diffs.size(0), device=diffs.device)
            labels[p.size(0):] = -1
            return F.hinge_embedding_loss(diffs, labels, margin)

        elif name == 'embedding_mse':
            diffs: Tensor = F.mse_loss(torch.cat([a, a]), torch.cat([p, n]), reduction='none')
            diffs = diffs.mean(-1)
            labels = torch.ones(diffs.size(0), device=diffs.device)
            labels[p.size(0):] = -1
            return F.hinge_embedding_loss(diffs, labels, margin)

        else:
            raise ValueError('Unknown loss function: {}'.format(name_))

    return loss
