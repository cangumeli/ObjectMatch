# import json
import os
import pickle
from collections import Counter, OrderedDict

import numpy as np
import quaternion
import torch
from tabulate import tabulate

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from pytorch3d.ops import corresponding_points_alignment

from nocpred.structures import Depths


_EvalResult = OrderedDict[str, dict[str, float]]


class ImageAlignmentEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, cfg):
        super().__init__()
        self.output_dir = cfg.OUTPUT_DIR
        self._metadata = MetadataCatalog.get(dataset_name)
        self._dataset_name = dataset_name
        self._index_to_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._id_to_name = self._metadata.thing_classes

        self.pred_file = os.path.join(self.output_dir, 'image_alignment_preds.pkl')

        # TODO: Move this to data?
        # with open('./scene_counts.json') as f:
        #    self._scene_counts = json.load(f)

    def reset(self):
        self._preds: dict[str, dict] = {}
        # self.evaluate(from_file=self.pred_file)
        # exit()

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            self._add_preds(input, output)

        '''if len(self._preds) > 10:
            self.evaluate()
            exit()'''

    def evaluate(self, from_file: str = '', print_every: int = 100) -> _EvalResult:
        if from_file != '':
            self.pred_file = from_file
            all_preds = self._load_preds()
        else:
            self._save_preds()
            all_preds = self._preds

        return self._eval_loop(all_preds, print_every)

    def _add_preds(self, input, output):
        if 'depth' not in input:
            raise ValueError('{} needs depth input!'.format(self.__class__.__name__))
        if 'pose' not in input:
            raise ValueError('{} needs camera pose input!'.format(self.__class__.__name__))

        file_name = input['file_name']
        file_name = file_name.replace(self._metadata.image_root, '')

        if 'instances' not in output:
            self._preds[file_name] = {'pose': input['pose'].numpy(), 'preds': []}
            return

        instances = output['instances'].to('cpu')
        instances = instances[torch.argsort(instances.scores, descending=True)]
        instances = instances[instances.scores > 0.5]
        if not len(instances):
            self._preds[file_name] = {'pose': input['pose'].numpy(), 'preds': []}
            return

        depth: Depths = input['depth']
        depth = depth.repeat(len(instances))
        depth_points = depth.back_project(input['intrinsic']).crop_and_resize_with_grids_from_boxes(
            instances.pred_boxes, crop_size=instances.pred_nocs.image_size[-1]
        )

        masks = instances.pred_nocs.masks() & depth_points.masks()
        depth_points_list: list[torch.Tensor] = depth_points.as_point_clouds(
            wrap_output=False, masks=masks
        )
        noc_points_list: list[torch.Tensor] = instances.pred_nocs.as_point_clouds(
            wrap_output=False, masks=masks
        )

        preds = []
        for i in range(len(instances)):
            class_name = self._id_to_name[self._index_to_id[instances.pred_classes[i].item()]]
            preds.append({
                'score': instances.scores[i].item(),
                'nocs': noc_points_list[i].numpy(),
                'depth': depth_points_list[i].numpy(),
                'category': class_name,
            })

        self._preds[file_name] = {'pose': input['pose'].numpy(), 'preds': preds}

    @staticmethod
    def _print(*args, **kwargs):
        print(*args, **kwargs, flush=True)

    def _save_preds(self):
        with open(self.pred_file, 'wb') as f:
            pickle.dump(self._preds, f)

    def _load_preds(self) -> dict:
        with open(self.pred_file, 'rb') as f:
            preds = pickle.load(f)
        return preds

    def _eval_loop(self, all_preds: dict[str, dict], print_every: int) -> _EvalResult:
        data_dicts: list[dict] = DatasetCatalog.get(self._dataset_name)
        corrects_per_class = Counter()
        total_per_class = Counter()
        self._print('Starting the eval loop...')
        for step, record in enumerate(data_dicts):
            file_name = record['file_name']
            if 'annotations' not in record:
                continue
            annots = record['annotations']

            if print_every > 0 and step % print_every == 0:
                self._print('{} / {}'.format(step, len(data_dicts)))

            # For scan2cad style constraints
            gt_counts_per_class = Counter(
                self._id_to_name[self._index_to_id[ann['category_id']]] for ann in annots
            )
            pred_counts_per_class = Counter()
            total_per_class.update(gt_counts_per_class)
            try:
                # import pdb; pdb.set_trace()
                pred_data = all_preds[file_name.replace(self._metadata.image_root, '')]
            except KeyError:
                pred_data = all_preds[file_name]
            if len(pred_data['preds']) == 0:
                continue
            # self._print('here!!!')
            preds: list = pred_data['preds']
            pose: np.ndarray = pred_data['pose']

            preds.sort(key=lambda x: x['score'], reverse=True)

            # scene = next(x for x in file_name.split('/') if x.startswith('scene'))
            inv_pose = np.linalg.inv(pose)
            # gt_counts_per_class = self._scene_counts[scene]

            covered = [False for _ in annots]
            for pred in preds:
                cat = pred['category']
                pred_counts_per_class[cat] += 1
                if pred_counts_per_class[cat] > gt_counts_per_class.get(cat, 0):
                    continue
                for index, gt in enumerate(annots):
                    if covered[index]:
                        continue
                    # import pdb; pdb.set_trace()
                    if self._id_to_name[gt['category_id']] != cat:
                        continue
                    t, q, s = gt['alignment']
                    scaled_nocs = torch.from_numpy(pred['nocs'] * np.asarray(s)).double()
                    depths = torch.from_numpy(pred['depth']).double()
                    trs = corresponding_points_alignment(scaled_nocs[None], depths[None])
                    pred_trs = [trs.T.squeeze().tolist(), trs.R.squeeze().t().numpy(), s]
                    pred_trs[1] = quaternion.as_float_array(
                        quaternion.from_rotation_matrix(pred_trs[1])
                    ).tolist()
                    gt_trs = decompose_mat4(inv_pose @ make_M_from_tqs(t, q, s))
                    if is_correct(tuple(pred_trs), gt_trs):
                        corrects_per_class[cat] += 1
                        covered[index] = True

        accuracies = {
            cat: corrects_per_class[cat] / total_per_class[cat] for cat in total_per_class
        }
        self._print('\nPer-Category Alignment Accuracies\n')
        results = [(k, 100 * np.round(v, decimals=4)) for k, v in accuracies.items()]
        results = sorted(results, key=lambda pair: pair[0])
        self._print(tabulate(
            results, tablefmt='github', headers=['Class', 'Accuracy'])
        )
        self._print()

        class_average = 100 * sum(accuracies.values()) / len(accuracies)
        instance_average = 100 * sum(corrects_per_class.values()) / sum(total_per_class.values())

        self._print(tabulate(
            [('Class Avg.', class_average), ('Instance Avg.', instance_average)],
            tablefmt='github',
            headers=['Task', 'Accuracy'],
        ))
        self._print()

        return OrderedDict({
            'image_alignment': {'class': class_average, 'instance': instance_average}
        })


def is_correct(
    pred_tqs: tuple[list[float], list[float], list[float]],
    gt_tqs: tuple[list[float], list[float], list[float]],
) -> bool:
    t0, q0, s0 = pred_tqs
    t1, q1, s1 = gt_tqs
    t_diff = np.linalg.norm(np.asarray(t0) - np.asarray(t1))
    r_diff = calc_rotation_diff(np.quaternion(*q0), np.quaternion(*q1))
    s_diff = 100.0 * np.abs(np.mean(np.asarray(s0) / np.asarray(s1)) - 1)
    return t_diff <= .2 and r_diff <= 20 and s_diff <= 20


def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:  # noqa
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def make_M_from_tqs(t: list[float], q: list[float], s: list[float]) -> np.ndarray:
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    return M


def decompose_mat4(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = M[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])
    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz
    q = quaternion.from_rotation_matrix(R[0:3, 0:3])
    q = quaternion.as_float_array(q)

    t = M[0:3, 3]

    return t, q, s
