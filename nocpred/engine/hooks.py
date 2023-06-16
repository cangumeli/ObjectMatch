import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from detectron2.data import (
    build_detection_train_loader,
    get_detection_dataset_dicts,
    MetadataCatalog,
)
from detectron2.engine import HookBase

from nocpred.data import Mapper


# TODO: Support distributed training
class ValStep(HookBase):
    def __init__(self, run_period: int = 10, window_size: int = 10):
        self.run_period = run_period
        self.window_size = window_size

    def before_train(self):
        cfg = self.trainer.cfg
        self._logger = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'eval_logs'))
        self._window: list[dict[str, float]] = []

        dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        mapper = Mapper(cfg, is_train=True, metadata=metadata, no_aug=True)

        self._sample_val_data = build_detection_train_loader(
            mapper=mapper,
            dataset=dataset,
            total_batch_size=batch_size,
            num_workers=num_workers,
        )
        self._sample_val_iter = iter(self._sample_val_data)

    def after_step(self):
        iter = self.trainer.iter + 1
        is_last = iter == self.trainer.max_iter
        if iter % self.run_period == 0 or is_last:
            model = self.trainer.model
            # NOTE: This assumes frozen batch norms
            # as in the case of most detectron2 models
            # FIXME: This will noise the internal loggings
            # such as mask accuracy and bbox statistics
            with torch.no_grad():
                datum = next(self._sample_val_iter)
                losses: dict[str, torch.Tensor] = model(datum)
                losses['total_loss'] = sum(losses.values())
                self._window.append({k: v.item() for k, v in losses.items()})

            if len(self._window) > self.window_size or is_last:
                values = defaultdict(list)
                for dt in self._window:
                    for k, v in dt.items():
                        if np.abs(v) > 0:
                            values[k].append(v)
                values = {k: np.median(v) for k, v in values.items()}
                for k, v in values.items():
                    self._logger.add_scalar(k, v, iter)
                self._logger.flush()
                self._window.clear()

    def after_train(self):
        self._logger.close()
