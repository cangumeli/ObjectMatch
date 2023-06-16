from copy import deepcopy
from typing import Any, Callable, Union
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnets
import torchvision.ops.misc as vops
import torchvision.transforms as T


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def __getitem__(self, _):
        return self

    def forward(self, x):
        return torch.empty(x.size(0), 0, 7, 7, device=x.device, dtype=x.dtype)


class ForegroundBackgroundEncoder(nn.Module):
    def __init__(
        self,
        depth: int = 18,
        freeze_at: int = 2,
        num_embedding: int = 2048,
        pdrop: float = 0.0,
        depth_input: bool = False,
        normal_input: bool = False,
        normalize_output: bool = False,
        freeze_bn: bool = True,
        large_fc: bool = False,
        global_pool: str = 'Max',
        bg_input: bool = True,
    ):
        super().__init__()
        resnet_fn = getattr(resnets, 'resnet' + str(depth))
        norm_layer = vops.FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        resnet: resnets.ResNet = resnet_fn(pretrained=True, norm_layer=norm_layer)
        resnet = resnet.requires_grad_(False)

        self.fg_net = nn.Sequential(
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
            ),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        if bg_input:
            self.bg_net = deepcopy(self.fg_net)
        else:
            self.bg_net = DummyModule()

        self.depth_input = depth_input
        if self.depth_input:
            self.fg_net_depth = deepcopy(self.fg_net)
            self.bg_net_depth = deepcopy(self.bg_net)

        self.normal_input = normal_input
        if self.normal_input:
            self.fg_net_normal = deepcopy(self.fg_net)
            self.bg_net_normal = deepcopy(self.bg_net)

        for i in range(freeze_at, len(self.fg_net)):
            self.fg_net[i].requires_grad_()
            self.bg_net[i].requires_grad_()
            if self.depth_input:
                self.fg_net_depth[i].requires_grad_()
                self.bg_net_depth[i].requires_grad_()
            if self.normal_input:
                self.fg_net_normal[i].requires_grad_()
                self.bg_net_normal[i].requires_grad_()

        last_block: Union[resnets.BasicBlock, resnets.Bottleneck] = resnet.layer4[-1]
        if isinstance(last_block, resnets.BasicBlock):
            out_channels = last_block.conv2.out_channels
        else:
            out_channels = last_block.conv3.out_channels

        if not large_fc:
            global_pool = getattr(nn, 'Adaptive{}Pool2d'.format(global_pool.capitalize()))
            self.output = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(out_channels * (1 + int(bg_input)), out_channels * 4, 2),  # 3, 3
                global_pool((1, 1)),
                nn.ReLU(True),
                nn.Flatten(),
                # nn.Dropout(pdrop),
                nn.Linear(out_channels * 4, num_embedding),
            )
        else:
            hiddens = out_channels * 4
            self.output = nn.Sequential(
                nn.Flatten(),
                nn.Linear(7 * 7 * out_channels * 2, hiddens),
                nn.ReLU(True),
                nn.Linear(hiddens, hiddens),
                nn.ReLU(True),
                nn.Linear(hiddens, num_embedding),
            )

        self.image_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.bool()),
        ])

        self.drop = nn.Dropout2d(p=pdrop)
        self.normalize_output = normalize_output

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data_list: list[dict]) -> torch.Tensor:
        mask = self._preprocess_mask(data_list)
        fg_images, bg_images = self._preprocess(data_list, mask)
        fg_embeds = self.fg_net(fg_images)
        bg_embeds = self.bg_net(bg_images)
        for modal in ('depth', 'normal'):
            if getattr(self, '{}_input'.format(modal)):
                fg_images, bg_images = self._preprocess(data_list, mask, modal)
                fg_embeds = fg_embeds + getattr(self, 'fg_net_' + modal)(fg_images)
                bg_embeds = bg_embeds + getattr(self, 'bg_net_' + modal)(bg_images)
        embeds = self.drop(torch.cat([fg_embeds, bg_embeds], dim=1))
        output = self.output(embeds)
        if self.normalize_output:
            output = F.normalize(output, dim=-1)
        return output

    def _preprocess_mask(self, data_list):
        masks = torch.stack([self.mask_transform(d['mask']) for d in data_list])
        return masks.to(self.device)

    def _preprocess(self, data_list: list[dict], masks: torch.Tensor, field: str = 'image'):
        images = torch.stack([self.image_transform(d[field]) for d in data_list])
        images = images.to(self.device)
        fg_image = images * masks
        bg_image = images * masks.logical_not()
        return fg_image, bg_image


ModelType = Union[ForegroundBackgroundEncoder, Callable[[Any], torch.Tensor]]


def build_model(cfg: dict) -> ModelType:
    model_cfg = cfg['MODEL']
    input_cfg = cfg['INPUT']
    model = ForegroundBackgroundEncoder(
        depth=model_cfg['DEPTH'],
        freeze_at=model_cfg['FREEZE_AT'],
        num_embedding=model_cfg['EMBED_DIM'],
        pdrop=model_cfg['DROPOUT'],
        depth_input=input_cfg['DEPTH'],
        normal_input=input_cfg['NORMAL'],
        normalize_output=model_cfg['NORMALIZE_EMBED'],
        freeze_bn=model_cfg['FREEZE_BN'],
        large_fc=model_cfg['LARGE_FC'],
        global_pool=model_cfg['GLOBAL_POOL_TYPE'],
        bg_input=input_cfg.get('BG_INPUT', True),
    )
    if torch.has_cuda:
        model = model.cuda()
    return model


if __name__ == '__main__':
    model = ForegroundBackgroundEncoder(depth_input=True)  # .cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    import os
    from dataset import CropDataset
    crop_data = CropDataset(
        './crops_assoc_no_filter_train_100.pkl',
        os.environ['HOME'] + '/Data/Resized400k/tasks/scannet_frames_25k',
        use_depth=True,
        use_normal=False,
        keep_ratio=False,
        box_scale=5.,
        normalize_depth=True,
    )
    data = []
    # for i in range(50):
    data.append(crop_data['scene0000_00', 0])
    result = model(data)
    from IPython import embed; embed()
