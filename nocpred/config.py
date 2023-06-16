import detectron2.model_zoo as model_zoo
from detectron2.config import CfgNode, get_cfg


def nocpred_config(file: str) -> CfgNode:
    cfg = get_cfg()
    # Set the default values
    cfg.MODEL.BACKBONE.FEATURE_AGGREGATION = 'mean'
    cfg.MODEL.BACKBONE.FEATURE_DROP = 0.0

    cfg.MODEL.ROI_HEADS.WITHOUT_NOCS = False

    cfg.MODEL.ROI_NOC_HEAD = CfgNode()
    cfg.MODEL.ROI_NOC_HEAD.NAME = "ConvUpsampleNOCHead"
    cfg.MODEL.ROI_NOC_HEAD.POOLER_RESOLUTION = 16
    cfg.MODEL.ROI_NOC_HEAD.CONDITION_ON_DEPTH = False
    cfg.MODEL.ROI_NOC_HEAD.CONDITION_ON_NORMAL = False
    cfg.MODEL.ROI_NOC_HEAD.NOC_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_LOSS = False
    cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_ROTATION_WEIGHT = 1.0
    cfg.MODEL.ROI_NOC_HEAD.PROCRUSTES_TRANSLATION_WEIGHT = 1.0
    cfg.MODEL.ROI_NOC_HEAD.USE_DECONV = True
    cfg.MODEL.ROI_NOC_HEAD.USE_PIXEL_SHUFFLE = False

    cfg.MODEL.ROI_NOC_HEAD.PREDICT_SCALE = False
    cfg.MODEL.ROI_NOC_HEAD.PREDICT_ALIGNMENT = False

    cfg.INPUT.COLOR = True
    cfg.INPUT.DEPTH = False
    cfg.INPUT.NORMAL = False

    cfg.INPUT.DEPTH_CMAP = 'jet'

    cfg.INPUT.DEPTH_DOWN = 1
    cfg.INPUT.NORMAL_DOWN = 1

    cfg.TEST.VAL_STEP = True
    cfg.TEST.F1_THRESH = 0.1
    cfg.TEST.AP_EVAL = True
    cfg.TEST.IMAGE_ALIGNMENT_EVAL = True
    cfg.TEST.NOC_AP_EVAL = True
    cfg.TEST.NOC_MASK_THRESH = 0.5

    # Load values from the actual file
    cfg.merge_from_file(file)

    # Load mask-rcnn backbone
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    )

    return cfg
