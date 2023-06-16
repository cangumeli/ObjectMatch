import os
import pickle
from math import ceil, floor
from typing import Optional, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset


class CropDataset(Dataset):
    def __init__(
        self,
        crop_data: Union[str, dict],
        image_root: str,
        size: int = 224,
        box_scale: float = 4.0,
        keep_ratio: bool = False,
        use_depth: bool = False,
        use_normal: bool = False,
        random_box_scale: Optional[float] = None,
        random_box_offset: Optional[float] = None,
        debug: bool = False,
        normalize_depth: bool = True,
    ):
        super().__init__()
        if isinstance(crop_data, str):
            with open(crop_data, 'rb') as f:
                self.crop_data: dict = pickle.load(f)
        else:
            self.crop_data: dict = crop_data

        self.image_root = image_root
        self.box_scale = box_scale
        self.size = size
        self.keep_ratio = keep_ratio
        self.scenes = sorted(self.crop_data.keys())
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.normalize_depth = normalize_depth
        self.random_box_scale = random_box_scale
        self.random_box_offset = random_box_offset
        self.debug = debug

    def __len__(self):
        return sum(len(v) for v in self.crop_data.values())

    def __getitem__(self, index: tuple[str, int]):
        scene, idx = index
        crop_data = self.crop_data[scene][idx]
        iid = crop_data['image_id']
        image_file = os.path.join(self.image_root, scene, 'color', '{}.jpg'.format(iid))
        image = Image.open(image_file)

        if self.use_depth or self.use_normal:
            depth = cv.imread(image_file.replace('color', 'depth').replace('.jpg', '.png'), -1)
            depth = depth.astype(np.float32) / 1000.
        else:
            depth = None
        
        mask = np.unpackbits(crop_data['mask']).reshape(crop_data['image_size'])

        return {
            **self.load_record(image, depth, mask, crop_data),
            'scene': scene,
            'iid': iid,
        }

    def load_record(
        self,
        image: Image.Image,
        depth: np.ndarray,
        mask: np.ndarray,
        crop_data: dict
    ):
        oid = crop_data['object_id']
        cid = crop_data['class_id']

        mask = Image.fromarray(mask.astype(np.uint8) * 255)

        crop = self._get_crop(image, crop_data['box'])
        image_crop = image.crop(crop)
        mask_crop = mask.crop(crop)

        if self.keep_ratio:
            image_crop, mask_crop = self._iso_resize(image_crop), self._iso_resize(mask_crop)
            image_crop = self._pad_image(image_crop, 255 // 2)
            mask_crop = self._pad_image(mask_crop, 0)
        else:
            image_crop = image_crop.resize((self.size, self.size))
            mask_crop = mask_crop.resize((self.size, self.size))

        depth_crop = None
        normal_crop = None

        if self.use_depth:
            cmap = plt.get_cmap('jet_r')
            # norm_factor = depth.max() if self.normalize_depth else 10.
            # colored_depth = cmap(depth / norm_factor)[..., :3]
            if self.normalize_depth:
                colored_depth = cmap(depth / depth.max())[..., :3]
            else:
                colored_depth = cmap(depth / 10)[..., :3]
            colored_depth = np.round(colored_depth * 255).astype(np.uint8)
            depth_crop = Image.fromarray(colored_depth).crop(crop)
            if self.keep_ratio:
                depth_crop = self._pad_image(self._iso_resize(depth_crop), 0)
            else:
                depth_crop = depth_crop.resize((self.size, self.size))

            # print('here')
            # depth_crop.save('stuff.jpg')

        if self.use_normal:
            z = cv.bilateralFilter(depth, -1, 6., 6.)
            zx = cv.Sobel(z, cv.CV_64F, 1, 0, ksize=5)
            zy = cv.Sobel(z, cv.CV_64F, 0, 1, ksize=5)
            normal = np.dstack((-zx, -zy, np.ones_like(z)))
            n = np.linalg.norm(normal, axis=2)
            normal[:, :, 0] /= n
            normal[:, :, 1] /= n
            normal[:, :, 2] /= n
            colored_normal = (normal + 1) / 2
            # colored_normal *= (depth > 0)[..., None]
            colored_normal = np.clip(np.round(colored_normal * 255).astype('uint8'), 0, 255)
            normal_crop = Image.fromarray(colored_normal).crop(crop)
            if self.keep_ratio:
                normal_crop = self._pad_image(self._iso_resize(normal_crop), 0)
            else:
                normal_crop = normal_crop.resize((self.size, self.size))

        if self.debug:
            image = np.asarray(image_crop)
            mask = np.asarray(mask_crop) > 0
            image[mask] = .5 * image[mask] + np.array([100, 0, 0])
            plt.imshow(image)
            # plt.show()
            # plt.imshow(mask_crop, cmap='Greys_r')
            plt.show()
            if depth_crop is not None:
                plt.imshow(depth_crop)
                plt.show()

        return {
            'image': image_crop,
            'mask': mask_crop,
            'depth': depth_crop,
            'normal': normal_crop,
            'class_id': cid,
            'object_id': oid,
        }

    def _get_crop(self, image: Image.Image, box: tuple[float, float, float, float]):
        w, h, cx, cy = self._get_whc_box(box)

        scale = self.box_scale
        if self.random_box_scale is not None:
            # Sample random noise around the box
            scale *= np.random.uniform(1 / self.random_box_scale, self.random_box_scale)
            if self.debug:
                print(scale)

        if self.random_box_offset is not None:
            w_offset = w * self.random_box_offset
            h_offset = h * self.random_box_offset
            w_offset = np.random.uniform(-w_offset, +w_offset)
            h_offset = np.random.uniform(-h_offset, +h_offset)
            box_ = np.array(box)
            box_[[0, 2]] += w_offset
            box_[[1, 3]] += h_offset
            w, h, cx, cy = self._get_whc_box(tuple(box_))
            if self.debug:
                print(w_offset, h_offset)

        w_, h_ = w * scale, h * scale
        x0_, y0_ = max(0, cx - w_ / 2), max(0, cy - h_ / 2)
        x1_, y1_ = min(image.width, cx + w_ / 2), min(image.height, cy + h_ / 2)
        return tuple(map(int, map(round, (x0_, y0_, x1_, y1_))))

    def _get_whc_box(self, box: tuple[float, float, float, float]):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        cx = (x1 + x0) / 2
        cy = (y1 + y0) / 2
        return w, h, cx, cy

    def _iso_resize(self, image: Image.Image):
        image_size = list(image.size)
        max_size = max(image_size)
        ind = image_size.index(max_size)
        ratio = self.size / max_size
        new_size = [0, 0]
        new_size[ind] = self.size
        new_size[1-ind] = round(image_size[1-ind] * ratio)
        return image.resize(new_size)

    # https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    def _pad_image(self, pil_img: Image.Image, color: int):
        width, height = pil_img.size
        w_diff = self.size - width
        h_diff = self.size - height
        right = ceil(w_diff / 2)
        left = floor(w_diff / 2)
        top = ceil(h_diff / 2)
        bottom = floor(h_diff / 2)
        new_width = width + right + left
        new_height = height + top + bottom
        if pil_img.mode == 'RGB':
            color = (color, color, color)
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result


class TripletDataset(Dataset):
    def __init__(self, triplet_file: str, crop_data: CropDataset):
        print('Loading {}...'.format(triplet_file))
        with open(triplet_file, 'rb') as f:
            triplets = pickle.load(f)

        self.scenes: list[str] = triplets['scenes']
        print('Allocating data...')
        triplets = triplets['triplets']
        self.triplets = np.array(triplets, dtype=np.uint32)
        # (len(triplets), 5), dtype=np.uint32)
        # print('Copying to numpy array')
        # np.copyto(self.triplets, triplets)
        self.crop_data = crop_data

        del triplets
        inter_file = triplet_file.replace('triplets_', 'triplets_inter_scene_')
        print(inter_file)
        if os.path.isfile(inter_file):
            with open(inter_file, 'rb') as f:
                triplets = pickle.load(f)
        # self.scenes: list[str] = triplets['scenes']
            print('Allocating inter-scene data...')
            triplets = triplets['triplets']
            self.inter_triplets = np.array(triplets, dtype=np.uint32)
        else:
            self.inter_triplets = np.array([])

    def __len__(self):
        return self.triplets.shape[0] + self.inter_triplets.shape[0]

    def __getitem__(self, index: int):
        if index < self.triplets.shape[0]:
            s1, a, p, s2, n = self.triplets[index]
        else:
            s1, a, p, s2, n = self.inter_triplets[index % self.triplets.shape[0]]

        scene1 = self.scenes[s1]
        scene2 = self.scenes[s2]

        src = self.crop_data[scene1, a]
        pos = self.crop_data[scene1, p]
        neg = self.crop_data[scene2, n]

        '''to_show = np.hstack([np.asarray(a['image']) for a in [src, pos, neg]])
        masks = np.hstack([np.asarray(a['mask']) for a in [src, pos, neg]]) > 0
        to_show[masks] = 0.5 * to_show[masks] + 0.5 * np.array([255, 0, 0])
        plt.imshow(to_show)
        plt.show()'''
        # print(scene, s, p, n)
        # print()
        return src, pos, neg

    @staticmethod
    def collate(data: list):
        src = [d[0] for d in data]
        pos = [d[1] for d in data]
        neg = [d[2] for d in data]
        return src + pos + neg


if __name__ == '__main__':
    data = CropDataset(
        './crops_assoc_no_filter_train_minimal_100.pkl',
        os.environ['HOME'] + '/Data/Resized400k/tasks/scannet_frames_25k',
        box_scale=2.5,
        use_depth=False,
        debug=True,
        use_normal=False,
        # random_box_offset=0.5,
        # random_box_scale=1.5,
    )

    for i in range(7, 100):
        print(i)
        for size in (1.5, 4, 5, 6):
            print(size)
            data.box_scale = size
            data['scene0001_00', i]

    exit()
    data['scene0470_00', 5]
    data['scene0339_00', 10]
    exit()
    print('Building dataset')
    trip_data = TripletDataset('triplets_assoc_no_filter_train_minimal_100.pkl', data)
    print('Building data-loader', flush=True)
    loader = DataLoader(trip_data, shuffle=True, collate_fn=lambda x: x, num_workers=4, batch_size=8)
    for datum in tqdm(loader):
        # pass
        exit()
