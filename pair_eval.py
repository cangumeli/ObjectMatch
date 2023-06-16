import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from optim.common import load_matrix
from optim.solver import test
from pairwise_backend import (
    KPConfig,
    NOCPredConfig,
    ObjectIDConfig,
    OptimConfig,
    PairwiseSolver,
    PairwiseVisualizer,
)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--test_folder', default='TestImages')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_folder', default='vis_pairs')
    parser.add_argument('--result_json', default='results.json')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--checkpoints', default='checkpoints')
    args = parser.parse_args(args)
    print(args)

    base_dir = os.path.abspath(args.test_folder)

    samples = []
    with open(args.file) as f:
        for line in f:
            if not line.startswith('scene'):
                continue
            scene, id0, id1 = line.split()
            samples.append((scene, id0, id1))

    match_files = run_sg(base_dir, samples)

    solver = PairwiseSolver(
        KPConfig(),
        NOCPredConfig(f'{args.checkpoints}/model_sym.pth', 'configs/NOCPred.yaml'),
        ObjectIDConfig(f'{args.checkpoints}/all_5'),
        OptimConfig(verbose=not args.silent),
    )
    if args.vis:
        visualizer = PairwiseVisualizer(args.vis_folder)

    results = []
    corrects = []
    for i, ((scene, id0, id1), match_file) in tqdm(
        enumerate(zip(samples, match_files)), dynamic_ncols=True, total=len(samples)
    ):
        print(f'{i + 1} / {len(samples)}: {scene} {id0} {id1}')
        
        color0 = os.path.join(base_dir, scene, 'color', f'{id0}.jpg')
        color1 = os.path.join(base_dir, scene, 'color', f'{id1}.jpg')

        record0 = solver.load_record(color0)
        record1 = solver.load_record(color1)
        match_data = np.load(match_file)
    
        pred_pose, gn_output, extras = solver(record0, record1, match_data, ret_extra_outputs=True)

        pose0 = load_matrix(os.path.join(base_dir, scene, 'pose', f'{id0}.txt'))
        pose1 = load_matrix(os.path.join(base_dir, scene, 'pose', f'{id1}.txt'))
        gt_pose = np.linalg.inv(pose0) @ pose1
        te, ae = test(pred_pose, gt_pose)
        print(te, ae)
        print()
        corrects.append(te <= 30 and ae <= 15)

        if args.vis:
            vis_name = f'{scene}_{id0}_{id1}'
            visualizer(vis_name, record0, record1, pred_pose, gn_output, extras)
            print()

        results.append((scene, id0, id1, te, ae))

    print(np.mean(corrects))

    with open(args.result_json, 'w') as f:
        json.dump(results, f)


def run_sg(base_dir, samples):
    print('Running super-glue...')
    sg_dir = os.path.abspath('./dump_features')

    sg_paths = []
    os.makedirs(sg_dir, exist_ok=True)

    total_new = 0
    temp_txt = 'temp_sg.txt'
    with open(temp_txt, 'w') as f:
        for scene, id0, id1 in samples:
            color0 = os.path.join(base_dir, scene, 'color', f'{id0}.jpg')
            color1 = os.path.join(base_dir, scene, 'color', f'{id1}.jpg')

            pose0 = load_matrix(os.path.join(base_dir, scene, 'pose', f'{id0}.txt'))
            pose1 = load_matrix(os.path.join(base_dir, scene, 'pose', f'{id1}.txt'))
            intr = load_matrix(os.path.join(base_dir, scene, 'intrinsics_depth.txt'))[:3, :3]

            sg_path = os.path.join(sg_dir, f'{scene}_{id0}_{id1}_matches.npz')
            sg_paths.append(sg_path)
            if os.path.isfile(sg_path):
                continue
            else:
                total_new += 1
                gt_pose_sg: np.ndarray = np.linalg.inv(np.linalg.inv(pose0) @ pose1)
                line = '{} {} 0 0 {} {} {}\n'.format(
                    color0,
                    color1,
                    ' '.join(map(str, intr.ravel().tolist())),
                    ' '.join(map(str, intr.ravel().tolist())),
                    ' '.join(map(str, gt_pose_sg.ravel().tolist())),
                )
                f.write(line)

    print(f'{len(samples) - total_new} / {len(samples)} keypoints are recovered from {sg_dir}')
    if total_new:
        os.system(
            f'{sys.executable} -u'
            + ' ../SuperGluePretrainedNetwork/match_pairs_scannet.py' 
            + f' --input_dir "/" --input_pairs {temp_txt} --output_dir {sg_dir}'
        )
    os.unlink(temp_txt)
    print(len(sg_paths))
    return sg_paths


if __name__ == '__main__':
    main(sys.argv[1:])
