from typing import Callable, Optional, Sequence, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import detectron2.layers as L
import detectron2.data.transforms as T
from detectron2.structures import Boxes, ImageList
from pytorch3d.ops import add_pointclouds_to_volumes
from pytorch3d.structures import Pointclouds, Volumes


class _Coordinates(object):
    def __init__(
        self,
        tensor,
        repeats: Optional[Union[int, Sequence[int]]] = None,
        eps: float = 1e-9,
        scale: Union[int, float] = 1.,
        offset: Optional[Union[int, float]] = None,
    ):
        # See detectron2.structures.BitMasks
        if not torch.is_tensor(tensor):
            device = torch.device('cpu')
            tensor = torch.as_tensor(tensor, device=device)

        assert tensor.ndim == 4

        if repeats is not None:
            # import pdb; pdb.set_trace()
            if tensor.size(0) > 1:
                tensor = tensor.repeat_interleave(repeats, dim=0)
            else:
                if not hasattr(repeats, '__len__'):
                    repeats = [repeats]
                else:
                    assert len(repeats) == 1
                tensor = tensor.expand(repeats[0], *tensor.shape[1:])

        self.tensor = tensor
        self.eps = eps
        self.scale = scale
        self.offset = offset

    def state(self, exclude: tuple[str] = ()):
        state = {
            'tensor': self.tensor,
            'eps': self.eps,
            'scale': self.scale,
            'offset': self.offset,
        }
        for e in exclude:
            del state[e]
        return state

    def state_wo_tensor(self):
        return self.state(exclude=('tensor',))

    def to(self, device: Union[torch.device, str], *args, **kwargs):
        state = self.state_wo_tensor()
        state['tensor'] = self.tensor.to(device, *args, *kwargs)
        return type(self)(**state)

    def repeat(self, repeats: Union[int, Sequence[int]]):
        return type(self)(repeats=repeats, **self.state())

    def unroll_instances(self, instance_img: Union[np.ndarray, torch.Tensor]):
        assert len(self) == 1, 'Instance unrolling only supports length==1'
        if isinstance(instance_img, np.ndarray):
            instance_img = torch.from_numpy(instance_img)
        ids: list[int] = instance_img.unique().cpu().tolist()
        ids.remove(0)
        coords = self.unroll_instances_with_given_ids(
            instance_img, ids, ret_valids=False
        )
        return coords, ids

    def unroll_instances_with_given_ids(
        self,
        instance_img: Union[torch.Tensor, np.ndarray],
        ids: Sequence[int],
        ret_valids: bool = True,
    ):
        assert len(self) == 1, 'Instance unrolling only supports length==1'
        assert 0 not in ids, '0 id must be reserved for background!'
        if isinstance(instance_img, np.ndarray):
            instance_img = torch.from_numpy(np.copy(instance_img))
        cls = type(self)

        coords = []
        valids: list[bool] = []
        for id in ids:
            mask = (instance_img == id)
            if ret_valids:
                valids.append(mask.any().item())
            tensor = mask * self.tensor
            coords.append(cls(tensor=tensor))

        coords = cls.cat(coords, **self.state_wo_tensor())
        if ret_valids:
            return coords, valids
        else:
            return coords

    @property
    def image_size(self) -> tuple[int, int, int]:
        return self.tensor.shape[1:]

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self):
        return self.tensor.dtype

    def clone(self):
        obj = type(self)(
            tensor=self.tensor.clone(), **self.state_wo_tensor()
        )
        if hasattr(obj, 'ndim'):
            obj.ndim = self.ndim
        return obj

    def __getitem__(self, item):
        tensor = self.tensor[item]
        if tensor.shape == self.tensor.shape[1:]:
            tensor = tensor[None]
        return type(self)(tensor=tensor, **self.state_wo_tensor())

    def __iter__(self):
        yield from self.tensor

    def __repr__(self):
        return self.__class__.__name__ + \
            "(tensor={}, eps={}, scale={}, offset={})".\
            format(str(self.tensor), str(self.eps), str(self.scale), str(self.offset))

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        return (self.tensor >= self.eps).flatten(1).any(dim=1)

    def masks(self) -> torch.Tensor:
        return (self.tensor.abs() >= self.eps).all(1, keepdim=True)

    def crop_and_resize(
        self,
        boxes_or_grids: torch.Tensor,
        crop_size: Optional[int] = None,
        use_interpolate: bool = False,
        use_grid: bool = False,
        wrap_output: bool = True,
    ):
        if use_grid:
            grid = boxes_or_grids.permute(0, 2, 3, 1)
            crops = F.grid_sample(self.tensor, grid, 'nearest', align_corners=False)
        else:
            boxes = boxes_or_grids
            assert crop_size is not None, 'crop_size argument is necessary when not using grid'
            assert len(boxes) == len(self), '{} != {}'.format(len(boxes), len(self))
            if use_interpolate:
                boxes = boxes.detach().cpu()
                tensor = self.tensor
                crops = []
                for i in range(boxes.size(0)):
                    [xs, ys, xe, ye] = boxes[i].round().int().tolist()
                    # print(i, [xs, ys, xe, ye], tensor.shape)
                    if xs >= xe or ys >= ye:
                        crop = torch.zeros(tensor.size(1), crop_size, crop_size, device=self.device)
                    else:
                        crop = F.interpolate(
                            tensor[i, :, ys:ye, xs:xe].unsqueeze(0),
                            size=(crop_size, crop_size),
                            mode='nearest',
                        ).squeeze(0)
                    crops.append(crop)
                crops = torch.stack(crops)
            else:
                device = self.device
                batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
                rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5
                rois = rois.to(device=device)
                crops = L.ROIAlign((crop_size, crop_size), 1.0, 1, aligned=True)(self.tensor, rois)

        return type(self)(tensor=crops, **self.state_wo_tensor()) if wrap_output else crops

    def crop_and_resize_with_grids_from_boxes(
        self,
        boxes: Boxes,
        crop_size: int,
        wrap_output: bool = True,
    ):
        image_size = self.image_size[1:][::-1]
        xy_grids = MeshGrids(image_size, batch_size=len(self))
        xy_grids = xy_grids.crop_and_resize_with_norm(
            boxes.tensor,
            use_interpolate=True,
            crop_size=crop_size,
            wrap_output=False,
        )[1]
        xy_grids = xy_grids.to(self.device)
        return self.crop_and_resize(xy_grids, use_grid=True, wrap_output=wrap_output)

    def as_point_clouds(
        self,
        wrap_output: Optional[bool] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Union[Pointclouds, list[torch.Tensor]]:
        if masks is None:
            masks = self.masks()

        pcds = []
        channel_size = self.tensor.size(1)
        for points, mask in zip(self.tensor.unbind(), masks.unbind()):
            points = points.squeeze(0)[:, mask.squeeze(0)].t()
            if points.numel() == 0:  # FIXME: This is hacky
                points = torch.zeros(1, channel_size, device=self.device)
            pcds.append(points)

        if wrap_output is None:
            wrap_output = channel_size == 3
        if wrap_output:
            feats = [torch.ones(p.size(0), 1, device=p.device) for p in pcds]
            return Pointclouds(pcds, features=feats)
        else:
            return pcds

    @classmethod
    def cat(
        cls,
        coords_list: Union[list['_Coordinates'], tuple['_Coordinates']],
        **params,
    ):
        assert isinstance(coords_list, (list, tuple))
        assert len(coords_list) > 0
        assert all(isinstance(coords, cls) for coords in coords_list)
        result = type(coords_list[0])(
            tensor=L.cat([c.tensor for c in coords_list], dim=0),
            **params,
        )
        return result

    @classmethod
    def decode(
        cls,
        image: np.ndarray,
        scale: Union[int, float],
        offset: Optional[float] = None,
        device='cpu',
        eps=1e-5,
        dtype=torch.get_default_dtype(),
    ):
        assert image.ndim in (2, 3), 'Unsupported image encoding'

        tensor = torch.tensor(image.astype(np.float32), dtype=dtype, device=device)
        tensor /= scale
        if offset is not None:
            tensor -= offset

        zero_mask = torch.from_numpy(image == 0)
        if zero_mask.ndim == 3:
            zero_mask = zero_mask.all(-1)
        tensor[zero_mask] = 0

        if tensor.ndim == 3:  # RGB encoding
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.ndim == 2:  # Single channel encoding
            tensor = tensor.view(1, 1, *tensor.size())

        tensor = tensor.to(device)

        return cls(tensor=tensor, eps=eps, scale=scale, offset=offset)
    
    def encode(self, target_type: np.dtype) -> np.ndarray:
        data = self.scale * self.tensor.cpu().numpy()
        if self.offset is not None:
            data = data + self.offset
        data = data.astype(target_type)
        return data


class NOCs(_Coordinates):
    # This doesn't support depths as voxel center is 0
    def voxelize(
        self,
        grid_size: Union[int, tuple[int, int, int]],
        normalize: bool = True,
        thresholded: bool = False,
        threshold: Optional[float] = None,
        mode: str = 'trilinear',
    ) -> torch.Tensor:

        points = self.as_point_clouds()
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size, grid_size)
        volume_size = (len(points), 1, *grid_size)
        volumes = Volumes(
            densities=torch.zeros(volume_size, device=points.device),
            voxel_size=(1 / (max(grid_size) - 1)),
            features=torch.zeros(volume_size, device=points.device),
        )
        try:
            volumes = add_pointclouds_to_volumes(points, volumes, mode=mode)
            voxels = volumes.densities()
        except ValueError:
            # if len(points) != 0:
            #    from IPython import embed; embed()
            assert len(points) == 0
            voxels = torch.zeros((0, 1, *grid_size), device=self.device)
            if thresholded:
                voxels = voxels.bool()
            return voxels
        # Normalize volumes
        # TODO: Support alternatives like softmax
        if normalize:
            voxels = voxels / voxels.sum((2, 3, 4), keepdim=True).clamp(1e-5)
        if thresholded:
            voxels = voxels >= (self.eps if threshold is None else threshold)
        return voxels

    @classmethod
    def from_file(
        cls, filename: str, transforms: T.Transform = T.NoOpTransform(), **kwargs
    ) -> 'NOCs':
        nocs = cv.imread(filename, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        assert nocs is not None, 'File {} not found!'.format(filename)
        nocs = transforms.apply_image(nocs)
        return cls.decode(nocs, scale=10000, offset=.5, **kwargs)


class Depths(_Coordinates):
    def back_project(
        self,
        intr: torch.Tensor,
        invert_intr: bool = True,
        wrap_output: bool = True,
    ) -> Union['DepthPoints', torch.Tensor]:
        batch_size, image_size = len(self), self.image_size[1:][::-1]
        grid = MeshGrids(image_size=image_size, batch_size=batch_size, device=self.device)
        if invert_intr:
            intr = intr.inverse()
        xy_grid = grid.tensor
        depth = self.tensor
        x = xy_grid[:, 0, :, :].flatten(1)
        y = xy_grid[:, 1, :, :].flatten(1)
        z = depth.flatten(1)
        points = intr @ torch.stack([x * z, y * z, z], dim=1)
        points = points.view(-1, 3, *depth.shape[-2:])
        return DepthPoints(tensor=points, eps=self.eps) if wrap_output else points

    def to_color(
        self,
        max_depth: float = 10.,
        quantize: bool = True,
        uint8: bool = True,
        bgr: bool = True,
        color_map: str = 'jet',
        down_sample: int = 1,
        wrap_output: bool = True,
    ) -> Union[ImageList, list[np.ndarray]]:
        c, h, w = self.image_size
        assert c == 1, 'Coloring is only supported for one channel'
        assert not uint8 or quantize, 'uint8 is only supported in quantized mode'
        cmap = plt.get_cmap(color_map)
        values = (self.tensor / max_depth).cpu()
        images: list[torch.Tensor] = []
        for image_tensor in values.unbind():
            image: np.ndarray = image_tensor.squeeze(0).numpy()
            if down_sample > 1:
                image = cv.bilateralFilter(image, -1, down_sample, down_sample)
                image = cv.resize(image, (w // down_sample, h // down_sample), cv.INTER_NEAREST)
            image = cmap(image)
            if quantize:
                image = np.round(image * 255)
            if uint8:
                image = image.astype(np.uint8)
            image = cv.cvtColor(image, cv.COLOR_RGBA2BGR if bgr else cv.COLOR_RGBA2RGB)
            images.append(torch.from_numpy(image).permute(2, 0, 1) if wrap_output else image)
        return ImageList(torch.stack(images), [(h, w) for _ in images]) if wrap_output else images

    @classmethod
    def from_file(
        cls, filename: str, transforms: Callable, scale=1000, **kwargs
    ) -> 'Depths':
        depth = cv.imread(filename, -1)
        assert depth is not None, 'File {} not found!'.format(filename)
        depth = transforms(depth)
        return cls.decode(depth, scale=scale, **kwargs)


class DepthPoints(_Coordinates):
    pass


class Normals(_Coordinates):
    @classmethod
    # TODO: make this proper pytorch
    def from_depths(cls, depths: Depths, down_sample: int = 1, **kwargs):
        dtype, device = depths.dtype, depths.device
        normals = []
        depth_batch = depths.tensor.cpu().unbind()
        for depth_tensor in depth_batch:
            depth: np.ndarray = depth_tensor.squeeze(0).numpy()
            depth = cv.bilateralFilter(depth, -1, 4., 4.)
            if down_sample > 1:
                _, h, w = depths.image_size
                depth = cv.resize(depth, (w // down_sample, h // down_sample), cv.INTER_NEAREST)
            depth = depth.astype(np.float64)

            # zy, zx = np.gradient(depth)
            zx = cv.Sobel(depth, cv.CV_64F, 1, 0, ksize=5)
            zy = cv.Sobel(depth, cv.CV_64F, 0, 1, ksize=5)
            normal = np.dstack((-zx, -zy, np.ones_like(depth)))
            n = np.linalg.norm(normal, axis=2)
            normal[:, :, 0] /= n
            normal[:, :, 1] /= n
            normal[:, :, 2] /= n
            normal = torch.from_numpy(normal)
            normals.append(normal.permute(2, 0, 1))
        return cls(tensor=torch.stack(normals).to(dtype=dtype, device=device), **kwargs)

    def to_color(
        self,
        quantize: bool = True,
        uint8: bool = True,
        bgr: bool = True,
        wrap_output: bool = True,
    ) -> Union[ImageList, list[np.ndarray]]:
        normals = self.tensor
        normals = (normals + 1) / 2
        if quantize:
            normals = torch.round(normals * 255)
        if uint8:
            normals = normals.to(dtype=torch.uint8)
        if bgr:
            normals = normals.flip(1)

        if wrap_output:
            return ImageList(normals, [self.image_size[-2:] for _ in range(len(self))])
        else:
            return [n.numpy().transpose(1, 2, 0) for n in self.tensor.cpu().unbind()]


class MeshGrids(_Coordinates):
    def __init__(
        self,
        image_size: Optional[tuple[int, int]] = None,
        batch_size: int = 1,
        device: Union[str, torch.device] = None,
        tensor: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if tensor is not None:
            super().__init__(tensor=tensor, **kwargs)

        elif batch_size == 0:
            tensor = torch.tensor([], device=device).view(0, 2, *image_size)
            super().__init__(tensor=tensor, **kwargs)

        else:
            # FIXME: image_size should be w, h and y, x should be x, y
            assert image_size is not None
            h, w = image_size
            y, x = torch.meshgrid(
                torch.linspace(0, w - 1, w, device=device),
                torch.linspace(0, h - 1, h, device=device),
            )
            tensor = torch.stack([x, y], dim=0).unsqueeze(0)
            if batch_size > 1:
                tensor = tensor.expand(batch_size, *tensor.shape[-3:])
            super().__init__(tensor=tensor, **kwargs)

    def crop_and_resize_with_norm(self, *args, **kwargs):
        res = self.crop_and_resize(*args, **kwargs)
        if not isinstance(res, torch.Tensor):
            res = res.tensor
        h, w = self.tensor.shape[-2:]
        res_n = 2 * res / torch.tensor(
            [w - 1, h - 1], device=self.device, dtype=self.tensor.dtype
        ).view(1, 2, 1, 1) - 1
        if kwargs.get('wrap_output', True):
            res, res_n = map(type(self), (res, res_n))
        return res, res_n
