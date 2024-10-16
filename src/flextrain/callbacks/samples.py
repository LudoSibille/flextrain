import json
import logging
import os
from collections import defaultdict
from glob import glob
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import torch
from mdutils.mdutils import MdUtils
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..trainer.utils import NumpyTorchEncoder, to_value
from ..types import Batch
from .callback import Callback

logger = logging.getLogger(__name__)


def export_image_2d(image: np.ndarray, path: str, data_range: Optional[Tuple[float, float]] = None) -> None:
    if data_range is None:
        data_range = image.min(), image.max()

    r = data_range[1] + 1e-5 - data_range[0]
    image_clipped = (255 * (np.clip(image, data_range[0], data_range[1]) - data_range[0]) / r).astype(np.uint8)
    if len(image_clipped.shape) == 3:
        # CHW -> HWC
        image_clipped = image_clipped.transpose((1, 2, 0))

    Image.fromarray(image_clipped).save(path + '.png')


def export_image_3d_slice_mosaic(
    image: np.ndarray,
    path: str,
    nb_slices: int = 9,
    data_range: Optional[Tuple[float, float]] = None,
) -> None:
    slices = np.arange(0, image.shape[0], image.shape[0] / nb_slices).astype(int)
    image_mosaic = make_grid(torch.from_numpy(image[slices]).unsqueeze(1), nrow=int(np.sqrt(nb_slices))).numpy()

    if data_range is None:
        data_range = image.min(), image.max()

    r = data_range[1] + 1e-5 - data_range[0]
    image_mosaic_clipped = (255 * (np.clip(image_mosaic, data_range[0], data_range[1]) - data_range[0]) / r).astype(np.uint8)
    Image.fromarray(image_mosaic_clipped.transpose((1, 2, 0))).save(path + '.png')


def export_single_batch(
    batch: Batch,
    basename: str,
    discard_keys: Sequence[str] = (),
    export_2d_raw: bool = True,
    export_3d_raw: bool = True,
    export_image_2d_fn: Callable[[np.ndarray, str], None] = export_image_2d,
    export_image_3d_fn: Callable[[np.ndarray, str], None] = export_image_3d_slice_mosaic,
    max_size_np_txt: int = 32,
) -> None:

    text_dict = {}
    for name, value in batch.items():
        if name in discard_keys:
            continue

        v = to_value(value)
        if isinstance(v, np.ndarray):
            v = v.squeeze()
            # 2D grayscale image OR 3D with RGB components
            if (len(v.shape) == 2 or (len(v.shape) == 3) and v.shape[0] == 3) and v.size > max_size_np_txt:
                base_data_path = basename + '_' + name
                export_image_2d_fn(v, base_data_path)
                if export_2d_raw:
                    np.save(base_data_path + '.npy', v)

            elif len(v.shape) == 3 and v.size > max_size_np_txt:
                base_data_path = basename + '_' + name
                export_image_3d_fn(v, base_data_path)
                if export_3d_raw:
                    np.save(base_data_path + '.npy', v)

            else:
                if v.size < max_size_np_txt:
                    text_dict[name] = v
                else:
                    base_data_path = basename + '_' + name
                    np.save(base_data_path + '.npy', v)
        else:
            text_dict[name] = v

    with open(basename + '.txt', 'w') as f:
        json.dump(text_dict, f, indent=3, cls=NumpyTorchEncoder)


class CallbackRecordSamples(Callback):
    """
    Export samples. This is mostly for validating data pipelines.
    """

    def __init__(
        self,
        output_dir_name: str = 'samples',
        split_names: Sequence[Literal['train', 'valid', 'test']] = ('train', 'valid'),
        nb_samples: int = 20,
        nb_repeat: int = 1,
        export_single_batch_fn: Callable[[Batch, str, Sequence[str]], None] = export_single_batch,
        discard_keys: Sequence[str] = (),
        **dataset_index_kwargs: Any,
    ):

        super().__init__()
        self.output_dir_name = output_dir_name
        self.output_path = None
        self.split_names = split_names
        self.nb_samples = nb_samples
        self.nb_repeat = nb_repeat
        self.dataset_index_kwargs = dataset_index_kwargs
        self.samples_exported = False
        self.export_single_batch_fn = export_single_batch_fn
        self.discard_keys = discard_keys

    def _export_sample(self, dataloader: DataLoader, split_name: str) -> None:
        nb_samples = min(len(dataloader.dataset), self.nb_samples)
        for sample_n in range(nb_samples):
            for repeat_n in range(self.nb_repeat):
                batch = dataloader.dataset.__getitem__(sample_n, **self.dataset_index_kwargs)
                basename = os.path.join(self.output_path, f'{split_name}-s{sample_n}-r{repeat_n}')
                self.export_single_batch_fn(batch, basename, self.discard_keys)

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.samples_exported:
            return
        else:
            self.samples_exported = True
            self.output_path = os.path.join(trainer.options.data.root_current_experiment, self.output_dir_name)
            os.makedirs(self.output_path, exist_ok=True)

        logger.info(f'exporting samples={self.output_path}')
        for split_name in self.split_names:
            data_loader = None
            if split_name == 'train':
                data_loader = trainer.train_dataloader
            elif split_name == 'valid':
                data_loader = trainer.val_dataloaders
            elif split_name == 'test':
                data_loader = trainer.test_dataloaders
            else:
                raise ValueError(f'unsupported split name={split_name}')

            if data_loader is None:
                logger.info(f'No dataloader for split={split_name}!')
                continue

            self._export_sample(data_loader, split_name)

    def make_markdown_report(self, md: MdUtils, base_level: int = 1) -> None:
        if self.output_path is None:
            return

        # TODO
        images = sorted([p for p in os.listdir(self.output_path) if '.png' in p])
