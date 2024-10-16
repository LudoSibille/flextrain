import os
from glob import glob
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

from ..types import Batch, Transform
from .utils import get_data_root


def to_rgb(image: Image) -> Image:
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class PairedImageDataset(Dataset):
    def __init__(
        self,
        root_A: str,
        root_B: str,
        transform: Optional[Transform] = None,
        paired: bool = True,
        image_ext: str = '.jpg',
        normalize_0_1: bool = False,
        normalize_minus1_1: bool = False,
    ):
        self.transform = transform
        self.paired = paired
        self.files_A = sorted(glob(os.path.join(root_A, f'*{image_ext}')))
        self.files_B = sorted(glob(os.path.join(root_B, f'*{image_ext}')))
        assert len(self.files_A) > 0, f'no file found at location={root_A}'
        assert len(self.files_B) > 0, f'no file found at location={root_B}'
        if paired:
            assert len(self.files_A) == len(self.files_B)

        self.normalize_0_1 = normalize_0_1
        self.normalize_minus1_1 = normalize_minus1_1
        assert int(normalize_minus1_1) + int(normalize_0_1) < 2, 'cannot apply 2 normalizations'

    def __getitem__(self, index: int) -> Batch:
        path_A = self.files_A[index % len(self.files_A)]
        image_A = Image.open(path_A)

        if not self.paired:
            path_B = self.files_B[np.random.randint(0, len(self.files_B) - 1)]
            image_B = Image.open(path_B)
        else:
            path_B = self.files_B[index % len(self.files_B)]
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != 'RGB':
            image_A = to_rgb(image_A)
        if image_B.mode != 'RGB':
            image_B = to_rgb(image_B)

        image_A_np = np.asarray(image_A).transpose((2, 0, 1))
        assert len(image_A_np.shape) == 3
        assert image_A_np.shape[0] == 3

        image_B_np = np.asarray(image_B).transpose((2, 0, 1))
        assert len(image_B_np.shape) == 3
        assert image_B_np.shape[0] == 3

        # normalize pixel range
        if self.normalize_0_1:
            image_A_np = image_A_np.astype(np.float32) / 255.0
            image_B_np = image_B_np.astype(np.float32) / 255.0

        if self.normalize_minus1_1:
            image_A_np = image_A_np.astype(np.float32) / 127.5 - 1
            image_B_np = image_B_np.astype(np.float32) / 127.5 - 1

        batch = {'A': image_A_np, 'B': image_B_np, 'A_path': path_A, 'B_path': path_B}
        if self.transform is not None:
            batch = self.transform(batch)

        return batch

    def __len__(self) -> int:
        return len(self.files_A)


def dataset_cyclegan(
    dataset_name: str,
    batch_size: int = 1,
    root: Optional[str] = None,
    num_workers: int = 4,
    normalize_0_1: bool = False,
    normalize_minus1_1: bool = True,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    persistent_workers: bool = True,
    A_dir: str = 'A',
    B_dir: str = 'B',
    transform_train: Optional[Transform] = None,
    transform_valid: Optional[Transform] = None,
    paired: bool = True,
    image_ext: str = '.jpg',
    splits: Sequence[str] = ('train', 'test'),
) -> Dict[str, DataLoader]:
    root = get_data_root(root)

    archive = f'http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip'
    download_and_extract_archive(url=archive, download_root=root, extract_root=root)

    dataloaders = {}
    for split in splits:

        if split == 'train':
            transform = transform_train
        else:
            transform = transform_valid

        ds = PairedImageDataset(
            root_A=os.path.join(root, dataset_name, split + A_dir),
            root_B=os.path.join(root, dataset_name, split + B_dir),
            transform=transform,
            paired=paired,
            image_ext=image_ext,
            normalize_0_1=normalize_0_1,
            normalize_minus1_1=normalize_minus1_1,
        )

        shuffle_split = shuffle and split == 'train'

        if max_samples is not None:
            assert not shuffle_split, 'incompatible!'
            # num_samples: restrict the number of samples per epoch
            sampler = torch.utils.data.RandomSampler(data_source=ds, num_samples=max_samples)
        else:
            if shuffle_split:
                sampler = torch.utils.data.RandomSampler(data_source=ds)
            else:
                sampler = torch.utils.data.SequentialSampler(data_source=ds)

        dl = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True,
        )

        if split == 'test':
            # this version of the dataset does not have the valid split
            # processed. Use the test set instead
            split = 'valid'

        dataloaders[split] = dl

    return {dataset_name: dataloaders}


def dataset_facades(**kwargs: Any) -> Dict[str, DataLoader]:
    return dataset_cyclegan(**kwargs, dataset_name='facades')


def dataset_apple2orange(**kwargs: Any) -> Dict[str, DataLoader]:
    return dataset_cyclegan(**kwargs, dataset_name='apple2orange')
