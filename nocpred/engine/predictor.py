from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from detectron2.modeling import build_model

from nocpred.data import Mapper


class Predictor:
    def __init__(
        self,
        cfg,
        model_path: str,
        depth_scale: float = 1000,
        resize_and_crop: float = 0.,
    ):
        self.model: nn.Module = build_model(cfg)
        self.model.load_state_dict(torch.load(model_path)['model'], strict=False)

        self.model.eval()
        self.model.requires_grad_(False)
        self.mapper = Mapper(
            cfg,
            is_train=False,
            input_only=True,
            depth_scale=depth_scale,
            resize_and_crop=resize_and_crop,
        )

    def load_record(
        self,
        file_name: str,
        intrinsic: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
        depth_file_name: Optional[str] = None,
    ) -> dict:

        if intrinsic is None:
            intrinsic = '/'.join(file_name.split('/')[:-2] + ['intrinsic_depth.txt'])
        if torch.is_tensor(intrinsic):
            intrinsic = intrinsic.cpu().numpy()
        if not isinstance(intrinsic, np.ndarray):
            with open(intrinsic) as f:
                intrinsic = np.array([[float(el) for el in line.strip().split()] for line in f])

        record = {'file_name': file_name, 'intrinsic': intrinsic}
        if depth_file_name is not None:
            record['depth_file_name'] = depth_file_name
        return self.mapper(record)

    @torch.no_grad()
    def __call__(self, inputs: Union[dict, list[dict]]):
        is_dict = isinstance(inputs, dict)
        if is_dict:
            inputs = [inputs]
        preds = self.model(inputs)
        if is_dict:
            preds = preds[0]
        return preds
