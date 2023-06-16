import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def register_scannet_split(
    name: str,
    data_dir: str,
    image_root: str,
    rendering_root: str,
    split: str,
    metadata: dict,
):
    json_file = os.path.join(data_dir, 'scannet_instances_{}.json'.format(split))

    extra_keys = ['alignment', 'object_id']
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, extra_keys)
    )

    # Fill lazy loading stuff
    DatasetCatalog.get(name)

    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type='coco',
        rendering_root=rendering_root,
        **metadata,
    )


def register_scannet(
    data_dir: str,
    image_root: str,
    rendering_root: str,
    metadata: dict,
    eval_only: bool = False,  # Performance hack for eval_only
) -> tuple[str, str]:

    names = []
    for split in ('train', 'val'):
        name = 'scannet_{}'.format(split)
        register_scannet_split(
            name=name,
            data_dir=data_dir,
            image_root=image_root,
            rendering_root=rendering_root,
            split=(split if not eval_only else 'val'),
            metadata=metadata,
        )
        names.append(name)

    return tuple(names)
