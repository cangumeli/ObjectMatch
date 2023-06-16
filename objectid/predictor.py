import os
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

import torch
from torch import Tensor

from .dataset import CropDataset
from .model import build_model


class Predictor:
    def __init__(self, output_dir: str):
        cfg = os.path.join(output_dir, 'config.yaml')
        with open(cfg) as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        self.cfg = cfg
        self.model = build_model(cfg).eval()

        ckpt = os.path.join(output_dir, 'last_model.pth')
        state = torch.load(ckpt)
        self.model.load_state_dict(state['model'])
        self._build_dataset()

    def _build_dataset(self):
        data_cfg = self.cfg['DATA']
        input_cfg = self.cfg['INPUT']
        self.dataset = CropDataset(
            crop_data={},
            image_root='',
            box_scale=data_cfg['BOX_SCALE'],
            keep_ratio=data_cfg['KEEP_RATIO'],
            use_depth=input_cfg['DEPTH'],
            normalize_depth=data_cfg['NORMALIZED_DEPTH'],
            use_normal=input_cfg['NORMAL'],
        )

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        depth: np.ndarray,
        boxes: list[list[float]],
        masks: np.ndarray,
        classes: list[int],
    ) -> Tensor:
        crop_data = self._load_crops(image, depth, boxes, masks, classes)
        return self.model(crop_data)

    def _load_crops(
        self,
        image,
        depth,
        boxes: list[list[float]],
        masks: np.ndarray,
        classes: list[int]
    ) -> list[dict]:
        records = []
        for oid, (box, mask, cat) in enumerate(zip(boxes, masks, classes)):
            crop_data = {
                'object_id': oid,
                'class_id': cat,
                'box': box,
            }
            records.append(self.dataset.load_record(image, depth, mask, crop_data))
        return records

    @staticmethod
    def associate(
        embeds0: Tensor,
        embeds1: Tensor,
        classes0: Tensor,
        classes1: Tensor,
        inf: float = 1000,
        thresh: float = 1.,
        top_n: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        dists = (embeds0[:, None] - embeds1[None]).square_().mean(-1)
        dists[classes0[:, None] != classes1[None]] = inf
        dists = dists.cpu().numpy()
        idx0, idx1 = linear_sum_assignment(dists)
        match_dists: np.ndarray = dists[idx0, idx1]

        mask = match_dists < thresh
        idx0, idx1, match_dists = idx0[mask], idx1[mask], match_dists[mask]
        if len(match_dists):
            order = np.argsort(match_dists)
            if top_n is not None:
                order = order[:top_n]
            idx0, idx1, match_dists = idx0[order], idx1[order], match_dists[order]

        return idx0, idx1, match_dists
