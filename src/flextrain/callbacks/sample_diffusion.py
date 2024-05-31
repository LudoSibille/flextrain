import logging
import math
import os
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torchvision.utils import make_grid

from ..diffusion.utils import catch_all_and_log
from ..metrics.fid import FID
from ..types import Batch

logger = logging.getLogger(__name__)


def unnorm_fn(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    return images


def unnorm_fid_fn(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    return images


class CallbackSample2dDiffusionModel(L.Callback):
    """
    Generate samples from the diffusion model
    """

    def __init__(
        self,
        sample_kwargs: Any,
        nb_samples: int = 1000,
        input_name: Optional[str] = None,
        input_conditioning_names: Optional[Union[str, Sequence[str]]] = None,
        sampling_folder: str = 'sampling',
        unnorm_fn: Callable[[torch.Tensor], torch.Tensor] = unnorm_fn,
        unnorm_conditioning_fn: Callable[[torch.Tensor], torch.Tensor] = unnorm_fn,
        unnorm_fid_fn: Callable[[torch.Tensor], torch.Tensor] = unnorm_fid_fn,
        deterministic_seed: bool = True,
        fid: Optional[FID] = None,
    ) -> None:
        super().__init__()
        self.sampling_folder = sampling_folder
        self.nb_samples = nb_samples
        self.sample_kwargs = sample_kwargs
        self.unnorm_fn = unnorm_fn
        self.input_name = input_name
        self.unnorm_conditioning_fn = unnorm_conditioning_fn

        self.input_conditioning_names = input_conditioning_names
        if input_conditioning_names is None:
            self.input_conditioning_names = ()
        elif isinstance(self.input_conditioning_names, str):
            self.input_conditioning_names = [self.input_conditioning_names]

        self.deterministic_seed = deterministic_seed
        self.seeds = []
        self.fid = fid
        self.unnorm_fid_fn = unnorm_fid_fn

    @catch_all_and_log
    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger.info('Sampling started!')
        nb_samples = 0

        def get_batch_iter() -> Iterator[Batch]:
            if trainer.val_dataloaders is not None:
                batch_iter = iter(trainer.val_dataloaders)
                logger.info('Using valid dataloader for the sampling!')
            else:
                assert trainer.train_dataloader is not None
                batch_iter = iter(trainer.train_dataloader)
                logger.warning('Using train dataloader for the sampling!')
            assert batch_iter is not None, 'no dataloader!'
            return batch_iter  # type: ignore

        true_batches = []
        image_sampled = []
        image_conditioning = []
        batch_iter = get_batch_iter()

        seeds = []
        batch_n = 0
        fid_features = []
        while nb_samples < self.nb_samples:
            logger.info(f'sampling={nb_samples}')

            if batch_iter is not None:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    # restart the sequence
                    batch_iter = get_batch_iter()
                    batch = next(batch_iter)
                true_batches.append(self.unnorm_fn(batch[self.input_name]).detach().cpu())

            conditioning = {}
            for name_n, name in enumerate(self.input_conditioning_names):
                value = batch.get(name)
                if isinstance(value, torch.Tensor):
                    value = value.to(pl_module.device)
                assert value is not None, f'conditioning={name}, not found!'
                conditioning[name] = value
                if name_n == 0 and len(value.shape) >= 4:
                    # record the first conditioning value
                    image_conditioning.append(self.unnorm_conditioning_fn(value).detach().cpu())

            # optional: specify the initial seeds
            # this is useful to understand the effect of
            # the training over time
            if self.deterministic_seed:
                if len(self.seeds) == 0:
                    x_T = torch.randn_like(batch[self.input_name])
                    seeds.append(x_T)
                else:
                    x_T = self.seeds[batch_n]
            else:
                x_T = torch.randn_like(batch[self.input_name])

            image = pl_module.sample(x_T=x_T, **self.sample_kwargs, **conditioning)
            image_sampled.append(self.unnorm_fn(image).detach().cpu())

            if self.fid is not None:
                # we need to normalize the image the same way the classifier
                # was trained on which may not necessarily be how
                # the diffusion model was preprocessed
                image_fid = self.unnorm_fid_fn(image)
                features = self.fid.calculate_features([image_fid], device=image_fid.device)
                fid_features.append(features.detach().cpu())

            batch_n += 1
            nb_samples += len(image)

        # record the first seed so we can restart from the same
        # random state next evaluation
        if len(seeds) > 0 and len(self.seeds) == 0:
            self.seeds = seeds
        logger.info('Sampling done!')
        output_folder = os.path.join(trainer.options.data.root_current_experiment, self.sampling_folder)
        os.makedirs(output_folder, exist_ok=True)

        if self.fid is not None:
            fid_features = torch.cat(fid_features, dim=0)
            fid_value = float(self.fid.calculate_fid_from_features(fid_features))
            print(f'fid_value={fid_value}')
            logger.info(f'fid_value={fid_value}')

        def export_grid(images: Sequence[torch.Tensor], path: str, apply_unnorm: bool = False) -> None:
            if not isinstance(images, torch.Tensor):
                images = torch.cat(images)
            if apply_unnorm:
                images = self.unnorm_conditioning_fn(images)

            images = images[: self.nb_samples]
            grid_sampled = make_grid(images, nrow=int(math.sqrt(self.nb_samples)))
            assert len(grid_sampled.shape) == 3, 'expecting channel x height x weidth'
            if grid_sampled.shape[0] != 3:
                # linearly sample 3 channels to be exported
                channels = np.linspace(0, grid_sampled.shape[0] - 1, 3, dtype=int)
                grid_sampled = grid_sampled[channels]

            Image.fromarray(grid_sampled.numpy().transpose((1, 2, 0))).save(path)

        export_grid(
            image_sampled, os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_sampled.png')
        )
        if len(image_conditioning) > 0:
            export_grid(
                image_conditioning,
                os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_conditioning.png'),
            )
        if batch_iter is not None:
            export_grid(
                true_batches, os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_trues.png')
            )
        if 'ref_img' in self.sample_kwargs:
            export_grid(
                self.sample_kwargs['ref_img'],
                os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_ref_img.png'),
                apply_unnorm=True,
            )

        logger.info(f'Exporting samples to={output_folder}')
