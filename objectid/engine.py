import json
import os
import random
from collections import defaultdict
from itertools import count
from typing import Any, Callable, Optional, Union

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import CropDataset, TripletDataset
from .loss import build_loss
from .model import build_model


class Engine:
    def __init__(
        self,
        config_file: str,
        image_root: str,
        train_crop_file: str,
        test_crop_file: str,
        train_triplet_file: str,
        test_triplet_file: Optional[str] = None,
        output_dir: str = './output',
        pair_file: Optional[str] = None,
    ):
        with open(config_file) as f:
            self.cfg = yaml.load(f, yaml.SafeLoader)

        seed = self.cfg['SEED']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(self.cfg['SEED'])

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.cfg, f)

        self.output_dir = output_dir
        self._build_train_data(image_root, train_crop_file, train_triplet_file)
        self._build_test_data(image_root, test_crop_file, test_triplet_file, pair_file)

        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_loggers()

        self.max_iters = self.cfg['SOLVER']['MAX_ITERS']
        self.backup_period = self.cfg['SOLVER']['BACKUP_PERIOD']
        self._step = 0

        print()
        print('Model:')
        print(self.model)

    def _build_train_data(self, image_root: str, crop_file: str, triplet_file: str):
        self.train_crop_data = CropDataset(
            crop_file,
            image_root,
            box_scale=self.cfg['DATA']['BOX_SCALE'],
            use_depth=self.cfg['INPUT']['DEPTH'],
            use_normal=self.cfg['INPUT']['NORMAL'],
            random_box_offset=self.cfg['DATA']['RANDOM_BOX_OFFSET'],
            random_box_scale=self.cfg['DATA']['RANDOM_BOX_SCALE'],
            normalize_depth=self.cfg['DATA']['NORMALIZED_DEPTH'],
            keep_ratio=self.cfg['DATA']['KEEP_RATIO'],
        )
        self.train_triplets = TripletDataset(triplet_file, self.train_crop_data)
        self.train_loader = DataLoader(
            self.train_triplets,
            batch_size=self.cfg['SOLVER']['BATCH_SIZE'],
            num_workers=self.cfg['SOLVER']['WORKERS'],
            collate_fn=TripletDataset.collate,
            shuffle=True,
            drop_last=True,
        )

    def _build_test_data(
        self,
        image_root: str,
        crop_file: str,
        triplet_file: Optional[str],
        pair_file: Optional[str],
    ):
        self.val_crop_data = CropDataset(
            crop_file,
            image_root,
            box_scale=self.cfg['DATA']['BOX_SCALE'],
            use_depth=self.cfg['INPUT']['DEPTH'],
            use_normal=self.cfg['INPUT']['NORMAL'],
            normalize_depth=self.cfg['DATA']['NORMALIZED_DEPTH'],
            keep_ratio=self.cfg['DATA']['KEEP_RATIO'],
        )
        self.val_step = self.cfg['SOLVER']['VAL_STEP']
        self.val_step_period = self.cfg['SOLVER']['VAL_STEP_PERIOD']
        self.val_triplets = TripletDataset(triplet_file, self.val_crop_data)
        self.eval_period = self.cfg['SOLVER']['EVAL_PERIOD']
        self.val_loader = DataLoader(
            self.val_triplets,
            batch_size=self.cfg['SOLVER']['BATCH_SIZE'],
            num_workers=self.cfg['SOLVER']['WORKERS'],
            collate_fn=TripletDataset.collate,
            shuffle=False,
            drop_last=False,
        )
        if self.val_step:
            self.val_step_loader = DataLoader(
                self.val_triplets,
                batch_size=self.cfg['SOLVER']['BATCH_SIZE'],
                num_workers=self.cfg['SOLVER']['WORKERS'],
                collate_fn=TripletDataset.collate,
                shuffle=True,
                drop_last=True,
            )
            self.val_iter = iter(self.val_step_loader)

        self.pair_data = None
        if pair_file is not None:
            with open(pair_file) as f:
                self.pair_data = json.load(f)['trajectories']

        # Testing metadata
        self._init_test_metadata()

    def _init_test_metadata(self):
        self.best_loss = np.inf
        self.best_acc = 0.
        self.bad_evals = 0
        self.bad_evals_to_decay = self.cfg['SOLVER']['BAD_EVALS_TO_DECAY']

        if self.pair_data is not None:
            self.best_pair_top1 = 0.
            self.best_pair_all = 0.

    @property
    def best_stuff(self):
        return {k: v for k, v in self.__dict__.items() if k.startswith('best_')}

    def _build_model(self):
        self.model = build_model(self.cfg)

    def _build_loss(self):
        self.loss = build_loss(self.cfg)

    def _build_optim(self):
        optim_cfg = self.cfg['SOLVER']['OPTIM']
        optim_type = getattr(torch.optim, optim_cfg['NAME'])
        self.optim: torch.optim.Optimizer = optim_type(
            self.model.parameters(), lr=optim_cfg['LR'], weight_decay=optim_cfg['WDECAY']
        )
        for pg in self.optim.param_groups:
            if 'momentum' in pg:
                print('Setting momentum...')
                pg['momentum'] = optim_cfg['MOMENTUM']

    def _build_loggers(self):
        self._loss_history = []
        self.logger = SummaryWriter(os.path.join(self.output_dir, 'train'))
        self._log_period = self.cfg['SOLVER']['LOG_PERIOD']
        if self.val_step:
            self._val_loss_history = []
            self.val_logger = SummaryWriter(os.path.join(self.output_dir, 'val'))

    def train(self):
        for e in count(start=1):
            print('Epoch {}'.format(e))
            # before_epoch = self._step
            for batch in self.train_loader:
                self.optim.zero_grad()
                src, pos, neg = self.model(batch).chunk(3)
                loss = self.loss(src, pos, neg)
                loss.backward()
                self.optim.step()

                self._step += 1

                self._loss_history.append(loss.item())
                if len(self._loss_history) % self._log_period == 0:
                    smooth_loss = np.median(self._loss_history)
                    self.logger.add_scalar('loss', smooth_loss, self._step)
                    self._loss_history.clear()
                    print('Step: {}, Loss: {}'.format(self._step, smooth_loss))

                if self._step % self.eval_period == 0:
                    # self.test()
                    self.test_pairs()

                if self._step % self.backup_period == 0:
                    self.backup_for_train('last_model.pth')

                if self.val_step and self._step % self.val_step_period == 0:
                    try:
                        batch = next(self.val_iter)
                    except StopIteration:
                        self.val_iter = iter(self.val_step_loader)
                        batch = next(self.val_iter)
                    with torch.no_grad():
                        self.model.eval()
                        src, pos, neg = self.model(batch).chunk(3)
                        loss = self.loss(src, pos, neg)
                        self._val_loss_history.append(loss.item())
                        self.model.train()
                    if len(self._val_loss_history) >= self._log_period:
                        smooth_loss = np.median(self._val_loss_history)
                        self.val_logger.add_scalar('loss', smooth_loss, self._step)
                        self._val_loss_history.clear()
                
                if self._step > self.max_iters:
                    print('Training ended')
                    return

    def test(self, backup: bool = True, log: bool = True):
        result = {}
        result['triplets'] = self.test_triplets(backup, log)
        if self.pair_data is not None:
            result['pairs'] = self.test_pairs(backup, log)
        return result

    @torch.no_grad()
    def test_triplets(
        self,
        backup: bool = True,
        log: bool = True,
        analysis_file: Optional[str] = None,
    ):
        print()

        self.model.eval()
        total_loss = 0.
        total_steps = 0
        total_corrects = 0
        should_analyze = analysis_file is not None
        analysis_data = []

        print('Running triplet evaluation at step: {}'.format(self._step))
        for i, val_batch in enumerate(self.val_loader):
            if i % 100 == 0:
                print('{} / {}'.format(i, len(self.val_loader)))
            src, pos, neg = self.model(val_batch).chunk(3)
            is_correct: Tensor = torch.norm(src - pos, dim=-1) < torch.norm(src - neg, dim=-1)

            numel = torch.numel(is_correct)
            if should_analyze:
                for k, cor in enumerate(is_correct.cpu().tolist()):
                    triplet_list = []
                    for datum in (val_batch[k], val_batch[k + numel], val_batch[k + 2 * numel]):
                        triplet_list.append({
                            'class_id': int(datum['class_id']),
                            'object_id': int(datum['object_id']),
                            'image_id': int(datum['image_id']),
                            'scene': datum['scene'],
                            'correct': bool(cor),
                        })
                    analysis_data.append(triplet_list)

            # from IPython import embed; embed()
            num_correct = torch.sum(is_correct)
            total_corrects += num_correct
            total_steps += src.size(0)
            total_loss += self.loss(src, pos, neg).item() * src.size(0)
            # break

        if should_analyze:
            analysis_file = os.path.join(self.output_dir, analysis_file)
            print('Dumping analysis results to {}'.format(analysis_file))
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f)

        average_loss = total_loss / total_steps
        print('Average loss is {}'.format(average_loss))
        acc = 100 * total_corrects / total_steps
        print('Accuracy is {}'.format(acc))

        if average_loss < self.best_loss:
            self.best_loss = average_loss
            print('**New best loss model**')
            if backup:
                self.backup_for_eval('best_loss_model.pth')

        if acc > self.best_acc:
            self.best_acc = acc
            print('**New best accuracy model**')
            if backup:
                self.backup_for_eval('best_acc_model.pth')
            self.bad_evals = 0
        else:
            self.bad_evals += 1

        if log:
            self.logger.add_scalar('eval_loss', average_loss, self._step)
            self.logger.add_scalar('accuracy', acc, self._step)

        if self.bad_evals >= self.bad_evals_to_decay:
            self.bad_evals = 0
            for pg in self.optim.param_groups:
                pg['lr'] = max(pg['lr'] * 0.1, 1e-8)
            print('Decaying learning rate to {}'.format(pg['lr']))

        result = {
            'accuracy': acc.item(),
            'loss': average_loss,
        }

        print('\nTriplet results\n')
        print(tabulate(list(result.items()), headers=['Metric', 'Average'], tablefmt='github'))
        print()

        return result

    @torch.no_grad()
    def test_pairs(
        self,
        backup: bool = True,
        log: bool = True,
        analysis_file: Optional[str] = None,
    ):
        assert self.pair_data is not None
        self.model.eval()
        scenes = self.val_crop_data.scenes
        top1_correct = []
        top2_correct = []
        top3_correct = []
        all_correct = []
        top1_plus_det_correct = []
        all_plus_det_correct = []

        should_analyze = analysis_file is not None
        if should_analyze:
            analysis_data = []

        for step, scene in enumerate(scenes, start=1):
            print('Scene {} / {}'.format(step, len(scenes)))
            pairs = self.pair_data[scene]
            scene_crop_data = self.val_crop_data.crop_data[scene]
            crop_by_image = defaultdict(list)
            for i, o in enumerate(scene_crop_data):
                crop_by_image[o['image_id']].append(i)

            for pair_idx, (i, j) in enumerate(pairs):
                crops_i = crop_by_image[i]
                crops_j = crop_by_image[j]
                oids_i = [scene_crop_data[k]['object_id'] for k in crops_i]
                oids_j = [scene_crop_data[k]['object_id'] for k in crops_j]

                if not len(oids_i) or not len(oids_j):
                    top1_plus_det_correct.append(False)
                    all_plus_det_correct.append(False)
                    continue

                batch = [self.val_crop_data[scene, idx] for idx in crops_i + crops_j]
                embeds = self.model(batch)
                res: tuple[Tensor, Tensor] = embeds.split([len(crops_i), len(crops_j)])
                zi, zj = res

                cats_i = np.array([scene_crop_data[b]['class_id'] for b in crops_i])
                cats_j = np.array([scene_crop_data[b]['class_id'] for b in crops_j])
                cmp: np.ndarray = cats_i[..., None] == cats_j[None]
                # print(zi.shape, zj.shape, cmp.shape)

                dists: np.ndarray = (zi[:, None] - zj[None]).square_().mean(-1).cpu().numpy()
                dists = cmp * dists + np.logical_not(cmp) * 1000

                i_idx, j_idx = linear_sum_assignment(dists)
                costs = dists[i_idx, j_idx]
                matches = []
                for m, n, c in zip(i_idx, j_idx, costs):
                    if c > 1:  # TODO: Make this configurable
                        continue
                    if cats_i[m] != cats_j[n]:
                        continue
                    matches.append((oids_i[m], oids_j[n], c))

                if not len(matches):
                    top1_correct.append(False)
                    all_correct.append(False)
                    continue

                matches.sort(key=lambda x: x[-1])
                is_correct = [oi == oj for oi, oj, _ in matches]
                if should_analyze:
                    analysis_data.append({
                        'scene': scene,
                        'index': pair_idx,
                        'matches': matches,
                        'is_correct': is_correct,
                    })

                top1_correct.append(is_correct[0])
                all_correct.append(all(is_correct))
                # if len(matches) >= 2:
                top2_correct.append(all(is_correct[:2]))
                # if len(matches) >= 3:
                top3_correct.append(all(is_correct[:3]))

        top1_plus_det_correct += top1_correct
        all_plus_det_correct += all_correct

        self.model.train()

        result = {
            'top1': np.mean(top1_correct) * 100,
            'top2': np.mean(top2_correct) * 100,
            'top3': np.mean(top3_correct) * 100,
            'all': np.mean(all_correct) * 100,
            'top1_det': np.mean(top1_plus_det_correct) * 100,
            'all_det': np.mean(all_plus_det_correct) * 100,
        }

        print('Pair results\n')
        print(tabulate(list(result.items()), headers=['Metric', 'Accuracy'], tablefmt='github'))
        print()

        if result['top1'] > self.best_pair_top1:
            self.best_pair_top1 = result['top1']
            print('**Best top1 model**')
            if backup:
                self.backup_for_eval('best_top1_model.pth')

        if result['all'] > self.best_pair_all:
            self.best_pair_all = result['all']
            print('**Best all model**')
            if backup:
                self.backup_for_eval('best_all_model.pth')

        if log:
            for k, v in result.items():
                self.logger.add_scalar('pairs/{}'.format(k), v, self._step)

        if should_analyze:
            analysis_file = os.path.join(self.output_dir, analysis_file)
            print('Dumping analysis to {}'.format(analysis_file))
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f)

        print()
        return result

    def backup_for_train(self, file_name: str):
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'step': self._step,
            **self.best_stuff,
        }, os.path.join(self.output_dir, file_name))

    def backup_for_eval(self, file_name: str):
        torch.save({
            'model': self.model.state_dict(),
            **self.best_stuff,
        }, os.path.join(self.output_dir, file_name))

    def resume(self, ckpt: Optional[str] = None, new_lr: Optional[float] = None):
        if ckpt is None:
            ckpt = os.path.join(self.output_dir, 'last_model.pth')
        state: dict[str, Any] = torch.load(ckpt)

        self._step = state.get('step', 0)
        if 'optim' in state:
            self.optim.load_state_dict(state['optim'])
        self.model.load_state_dict(state['model'])

        # For legacy backups!
        self.best_loss = state.get('loss', np.inf)
        self.best_acc = state.get('accuracy', 0)

        if new_lr is not None:
            for pg in self.optim.param_groups:
                pg['lr'] = new_lr

        # This overrides legacy values for new backups!
        for k, v in state.items():
            if k.startswith('best_') and hasattr(self, k):
                setattr(self, k, v)
