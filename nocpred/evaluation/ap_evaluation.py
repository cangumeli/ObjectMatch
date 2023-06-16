import json
import os
import pickle
from collections import defaultdict, OrderedDict
from itertools import chain
from typing import Any

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, Instances, pairwise_iou
from pytorch3d.ops import knn_gather, knn_points
from pytorch3d.structures import Pointclouds

from nocpred.structures import MeshGrids, NOCs


_EvalResult = OrderedDict[str, dict[str, float]]


class APEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, cfg, thresh=0.5):
        super().__init__()
        self.ap_fields = ('box', 'mask')
        self.output_dir = cfg.OUTPUT_DIR
        self.pred_file = os.path.join(self.output_dir, 'per_frame_preds.pkl')
        self.thresh = thresh
        self.f1_thresh = cfg.TEST.F1_THRESH

        self._metadata = MetadataCatalog.get(dataset_name)
        self._index_to_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._id_to_name = self._metadata.thing_classes

        self.noc_ap_eval = cfg.TEST.NOC_AP_EVAL
        if self.noc_ap_eval:
            self.ap_fields += ('noc',)

    def reset(self):
        self.preds = {}
        self.step = 0
        # self.evaluate(self.output_dir + '/per_frame_preds.pkl')
        # exit()

    def process(self, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]):
        self.step += 1
        for input, output in zip(inputs, outputs):
            self._add_preds(input, output)
        '''if self.step > 50:
            self.evaluate(); exit()'''

    def evaluate(self, from_file: str = '', print_every: int = 100) -> _EvalResult:
        # Load predictions
        if from_file != '':
            self.pred_file = from_file
            all_preds = self._load_preds()
        else:
            self._save_preds()
            all_preds = self.preds

        # Collect results
        per_class_ap_data = self._eval_loop(all_preds, print_every)

        # Compute the global (instance) APs
        gAPs = {}
        for f, ap_data in per_class_ap_data.items():
            ap_values = ap_data.values()
            scores = chain(*(v['scores'] for v in ap_values))
            labels = chain(*(v['labels'] for v in ap_values))
            scores, labels = map(torch.as_tensor, map(list, (scores, labels)))
            npos = sum(v['npos'] for v in ap_values)
            gAPs[f] = np.round(
                compute_ap(scores, labels, npos).item() * 100,
                decimals=2,
            ).item()

        # Compute per-category APs
        per_class_aps = {f: {} for f in per_class_ap_data.keys()}
        for f, ap_dict in per_class_aps.items():
            for cat, ap_data in per_class_ap_data[f].items():
                ap_dict[cat] = compute_ap(
                    torch.as_tensor(ap_data['scores']),
                    torch.as_tensor(ap_data['labels']),
                    ap_data['npos']
                ).item()

        # Average and report category APs
        mAPs = {}
        for f, v in per_class_aps.items():
            self.print('\nPer-Category Results for {}\n'.format(f.capitalize()))
            tab_data = list(v.items())
            tab_data = [
                (k, np.round(ap * 100, decimals=2)) for k, ap in tab_data
            ]
            tab_data.sort(key=lambda x: x[0])
            self.print(tabulate(
                tab_data,
                tablefmt='github',
                headers=['Category', 'AP'],
            ))
            mAPs[f] = np.round(
                np.mean([ap for _, ap in tab_data]),
                decimals=2,
            ).item()
            self.print()

        # Report and return mAPs
        for name, result in zip(['Mean', 'Instance'], [mAPs, gAPs]):
            self.print('\n{} APs Per Task\n'.format(name))
            self.print(tabulate(
                [(k.capitalize(), v) for k, v in result.items()],
                tablefmt='github',
                headers=['Task', 'mAP'],
            ))
            self.print()

        return OrderedDict({f: {'mAP': mAPs[f], 'gAP': gAPs[f]} for f in per_class_aps.keys()})

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs, flush=True)

    def _add_preds(self, input, output):
        instances: Instances = output['instances']  # .to('cpu')

        # grid_size = (32, 32, 32)  # TODO: Make this configurable
        pred_nocs: NOCs = instances.pred_nocs
        pred_xy_grids: MeshGrids = instances.pred_xy_grids
        noc_masks = pred_nocs.masks()
        noc_pcds = pred_nocs.as_point_clouds(masks=noc_masks)
        xy_pcds = pred_xy_grids.as_point_clouds(masks=noc_masks)

        ''' noc_occupancies: torch.Tensor = pred_nocs.voxelize(
            grid_size, normalize=False, thresholded=True, threshold=1
        ).cpu()'''

        # instances.remove('pred_xy_grids')
        # instances.remove('pred_nocs')
        instances = instances.to('cpu')

        objects = []
        for i in range(len(instances)):
            # Register trivial predictions
            datum = {
                'score': instances.scores[i].item(),
                'bbox': instances.pred_boxes.tensor[i].cpu().tolist(),
            }

            # Register classes
            class_name = self._id_to_name[self._index_to_id[instances.pred_classes[i].item()]]
            datum['category'] = class_name

            # Register rle mask
            mask = instances.pred_masks[i].numpy().squeeze()
            rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            datum['segmentation'] = rle

            # Register NOCs
            '''noc_indices = np.where(noc_occupancies[i].flatten(1).numpy())[0].tolist()
            datum['nocs'] = {
                'indices': noc_indices,
                'grid_size': grid_size,
                'roi_size': pred_nocs.image_size[-1],
            }'''
            datum['nocs'] = noc_pcds.points_list()[i].cpu().numpy()
            datum['noc_roi_size'] = pred_nocs.image_size[-1]

            datum['xy_grids'] = xy_pcds[i].cpu().numpy()

            # Add predictions
            objects.append(datum)

        parts = '/'.join(input['file_name'].split('/')[-3:])
        self.preds[parts] = objects

    def _save_preds(self):
        with open(self.pred_file, 'wb') as f:
            pickle.dump(self.preds, f)

    def _load_preds(self) -> dict:
        with open(self.pred_file, 'rb') as f:
            preds = pickle.load(f)
        return preds

    def _parse_data_json(self) -> tuple[dict[str, int], dict[int, dict]]:
        json_file = self._metadata.json_file
        with open(json_file) as f:
            gt = json.load(f)

        categories = gt['categories']
        file_to_id = {i['file_name']: i['id'] for i in gt['images']}
        file_to_id = {'/'.join(k.split('/')[-3:]): v for k, v in file_to_id.items()}

        id_to_annots = {id: [] for id in file_to_id.values()}
        for annot in gt['annotations']:
            annot['category'] = next(
                c['name'] for c in categories
                if c['id'] == annot['category_id']
            )
            id_to_annots[annot['image_id']].append(annot)

        return file_to_id, id_to_annots

    def _eval_loop(self, all_preds, print_every=500) -> dict:
        per_class_ap_data = {
            f: defaultdict(lambda: {'scores': [], 'labels': [], 'npos': 0})
            for f in self.ap_fields
        }
        file_to_id, id_to_annots = self._parse_data_json()

        self.print('\nStarting per-frame evaluation')
        covered_ids = set()
        for n, file_name in enumerate(all_preds.keys()):
            if print_every > 0 and n % print_every == 0:
                self.print('Frame: {}/{}'.format(n, len(all_preds)))

            preds = all_preds[file_name]
            try:
                annots = id_to_annots[file_to_id[file_name]]
            except KeyError:  # 400k
                img_name = file_name.split('/')[-1]
                new_img_name = '0' * (10 - len(img_name)) + img_name
                file_name = file_name.replace(img_name, new_img_name)
                annots = id_to_annots[file_to_id[file_name]]

            covered_ids.add(file_to_id[file_name])

            if not len(annots):
                for pred in preds:
                    for f in self.ap_fields:
                        ap_data = per_class_ap_data[f][pred['category']]
                        ap_data['labels'].append(0.)
                        ap_data['scores'].append(pred['score'])
                continue

            for annot in annots:
                for f in self.ap_fields:
                    per_class_ap_data[f][annot['category']]['npos'] += 1

            if not len(preds):
                continue

            preds = sorted(preds, key=lambda x: x['score'], reverse=True)

            # Box IOUs
            pred_boxes = Boxes([p['bbox'] for p in preds])
            gt_boxes = [gt['bbox'] for gt in annots]
            gt_boxes = Boxes([
                BoxMode.convert(
                    box,
                    from_mode=BoxMode.XYWH_ABS,
                    to_mode=BoxMode.XYXY_ABS,
                ) for box in gt_boxes
            ])
            box_ious = pairwise_iou(pred_boxes, gt_boxes)

            # Mask IOUs
            raw_gt_masks = gt_masks = mask_util.decode([gt['segmentation'] for gt in annots])
            pred_masks = mask_util.decode([p['segmentation'] for p in preds])

            pred_masks = pred_masks.reshape(-1, len(preds)).T[:, None, :]
            gt_masks = gt_masks.reshape(-1, len(annots)).T[None, :, :]

            unions = np.sum(np.logical_or(pred_masks, gt_masks), axis=-1)
            inters = np.sum(np.logical_and(pred_masks, gt_masks), axis=-1)
            mask_ious = inters / unions

            fields = ['box', 'mask']
            field_ious = [box_ious, mask_ious]

            # NOC IOUs
            if self.noc_ap_eval:
                noc_ious = self._compute_noc_ious(
                    file_name,
                    preds,
                    annots,
                    gt_boxes,
                    box_ious,
                    raw_gt_masks,
                )
                fields.append('noc')
                field_ious.append(noc_ious)

            # Collect AP labels and scores
            for field, ious in zip(fields, field_ious):
                covered = [False for _ in annots]
                for i in range(len(preds)):
                    matched = False
                    for j in range(len(annots)):
                        if covered[j]:
                            continue
                        category = preds[i]['category']
                        if category != annots[j]['category']:
                            continue
                        if ious[i, j] > self.thresh:
                            covered[j] = True
                            matched = True
                            break
                    ap_data = per_class_ap_data[field][category]
                    ap_data['scores'].append(preds[i]['score'])
                    ap_data['labels'].append(float(matched))

        # from IPython import embed; embed(); exit()
        excluded = set(id_to_annots.keys()).difference(covered_ids)
        self.print('{} / {} excluded'.format(len(excluded), len(id_to_annots)))
        for id in excluded:
            annots = id_to_annots[id]
            for annot in annots:
                for f in self.ap_fields:
                    per_class_ap_data[f][annot['category']]['npos'] += 1

        return per_class_ap_data

    def _compute_noc_ious(
        self,
        file_name: str,
        preds: list[dict],
        annots: list[dict],
        gt_boxes: Boxes,
        box_ious: torch.Tensor,
        raw_gt_masks: np.ndarray,
    ):
        # NOC AP
        device = 'cuda' if torch.has_cuda else 'cpu'
        noc_file = file_name\
            .replace('color', 'noc')\
            .replace('tasks/scannet_frames_25k/', '')\
            .replace('.jpg', '.png')
        noc_file = os.path.join(self._metadata.rendering_root, noc_file)

        # mask_imgs = raw_gt_masks
        gt_nocs = NOCs.from_file(noc_file, device=device).repeat(len(annots))
        raw_gt_masks = torch.from_numpy(raw_gt_masks.transpose(2, 0, 1))\
            .bool().to(device).unsqueeze(1)
        gt_nocs.tensor = gt_nocs.tensor * raw_gt_masks

        crop_size = preds[0]['noc_roi_size']
        gt_nocs = gt_nocs.crop_and_resize_with_grids_from_boxes(gt_boxes.to(device), crop_size)
        gt_points = gt_nocs.as_point_clouds()
        pred_points = Pointclouds([torch.from_numpy(p['nocs']) for p in preds]).to(device)

        aps = torch.zeros(len(preds), len(annots))
        covered = [False for _ in annots]
        for i, (pred, pred_pcd) in enumerate(zip(preds, pred_points.points_list())):
            for j, (annot, gt_pcd) in enumerate(zip(annots, gt_points.points_list())):
                if covered[j]:
                    continue
                if pred['category'] != annot['category']:
                    continue
                if box_ious[i, j] <= self.thresh:
                    continue
                if pred_pcd.numel() / 3 < 10 or gt_pcd.numel() / 3 < 10:
                    continue
                metrics = compute_sampling_metrics(
                    pred_points=pred_pcd.reshape(1, -1, 3),  # * 10,
                    pred_normals=None,
                    gt_points=gt_pcd.reshape(1, -1, 3),  # * 10,
                    gt_normals=None,
                    thresholds=(self.f1_thresh,),
                    eps=1e-9,
                )
                metric_key = next(k for k in metrics if 'F1@' in k)
                score = metrics[metric_key].item() / 100
                aps[i, j] = score
                if score > self.thresh:
                    covered[j] = True
                    break

        return aps


def compute_ap(scores, labels, npos, device=None):
    if device is None:
        device = scores.device

    if len(scores) == 0:
        return torch.tensor(0.0)
    tp = labels == 1
    fp = labels == 0
    sc = scores
    assert tp.size() == sc.size()
    assert tp.size() == fp.size()
    sc, ind = torch.sort(sc, descending=True)
    tp = tp[ind].to(dtype=torch.float32)
    fp = fp[ind].to(dtype=torch.float32)
    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)

    # # Compute precision/recall
    rec = tp / npos   # tp + fp
    # import pdb; pdb.set_trace()
    prec = tp / (fp + tp)
    ap = xVOCap(rec, prec)

    # import pdb; pdb.set_trace()
    return torch.as_tensor(ap)


def xVOCap(rec, prec):

    z = rec.new_zeros((1))
    o = rec.new_ones((1))
    mrec = torch.cat((z, rec, o))
    mpre = torch.cat((z, prec, z))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # import pdb; pdb.set_trace()

    I = (mrec[1:] != mrec[0:-1]).nonzero()[:, 0] + 1
    # import pdb; pdb.set_trace()
    ap = 0
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
    '''
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    '''
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics['Chamfer-L2'] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics['NormalConsistency'] = normal_dist
        metrics['AbsNormalConsistency'] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics['Precision@%f' % t] = precision
        metrics['Recall@%f' % t] = recall
        metrics['F1@%f' % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics
