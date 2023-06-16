import os

import torch.multiprocessing as mp

from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    MetadataCatalog,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils.events import CommonMetricPrinter, TensorboardXWriter

from nocpred.data import Mapper
from nocpred.engine.checkpoint import configure_checkpointer
from nocpred.engine.hooks import ValStep
from nocpred.evaluation import APEvaluator, ImageAlignmentEvaluator


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        mp.set_sharing_strategy('file_system')
        super().__init__(cfg)
        configure_checkpointer(self.checkpointer)

    def build_writers(self):
        output_dir = os.path.join(self.cfg.OUTPUT_DIR, 'train_logs')
        return [CommonMetricPrinter(self.max_iter), TensorboardXWriter(output_dir)]

    def build_hooks(self):
        hooks = super().build_hooks()
        if self.cfg.TEST.VAL_STEP:
            hooks.append(ValStep())
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = Mapper(cfg, is_train=True, metadata=metadata)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        metadata = MetadataCatalog.get(dataset_name)
        mapper = Mapper(cfg, is_train=False, metadata=metadata)
        return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        ap_eval = cfg.TEST.AP_EVAL
        image_alignment_eval = cfg.TEST.IMAGE_ALIGNMENT_EVAL
        evaluators = []
        if ap_eval:
            evaluators.append(APEvaluator(dataset_name, cfg))
        if image_alignment_eval:
            evaluators.append(ImageAlignmentEvaluator(dataset_name, cfg))

        if len(evaluators) == 0:
            raise ValueError('No valid evaluator type specified!')
        elif len(evaluators) == 1:
            return evaluators[0]
        else:
            return DatasetEvaluators(evaluators)
