import logging
import math
from typing import Callable, Dict, Optional
import torch
import lightning as L
import numpy as np
import os
from PIL import Image
from lightning.pytorch.utilities import rank_zero_only
from torchvision.utils import make_grid
from ..trainer.utils import transfer_batch_to_device


logger = logging.getLogger(__name__)


def unnorm_fn(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(0, 1) * 255).type(torch.uint8)
    return images


def get_batch_iter(trainer):
    if trainer.val_dataloaders is not None:
        batch_iter = iter(trainer.val_dataloaders)
    else:
        batch_iter = iter(trainer.train_dataloader)
    assert batch_iter is not None, 'no dataloader!'
    return batch_iter


def export_grid(images, path, nb_samples, unnorm_conditioning_fn=None, save_as_numpy=False):
    if not isinstance(images, torch.Tensor):
        images = torch.cat(images)
    if unnorm_conditioning_fn is not None:
        images = unnorm_conditioning_fn(images)

    images = images[:nb_samples]
    grid_sampled = make_grid(images, nrow=int(math.sqrt(nb_samples)))
    if save_as_numpy:
        np.save(path.replace('.png', '') + '.npy', grid_sampled.numpy())
    else:
        Image.fromarray(grid_sampled.numpy().transpose((1, 2, 0))).save(path)


class CallbackAutoenderRecon(L.Callback):
    def __init__(
            self, 
            input_name: str,
            nb_samples: int = 10,
            sampling_folder: str = 'recon', 
            unnorm_truth_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = unnorm_fn,
            unnorm_output_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = unnorm_fn,
            encode_kwargs: Dict = {},
            decode_kwargs: Dict = {},
            deterministic_seed: bool = True,
            save_numpy: bool = False,
            ) -> None:
        
        super().__init__()
        self.nb_samples = nb_samples
        self.input_name = input_name
        
        self.sampling_folder = sampling_folder
        self.unnorm_truth_fn = unnorm_truth_fn if unnorm_truth_fn is not None else lambda x:x
        self.unnorm_output_fn = unnorm_output_fn if unnorm_output_fn is not None else lambda x:x
        self.deterministic_seed = deterministic_seed

        self.encode_kwargs = encode_kwargs
        self.decode_kwargs = decode_kwargs

        self.seeds = []
        self.save_numpy = save_numpy

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        logger.info('Sampling started!')
        nb_samples = 0
        
        true_batches = []
        image_sampled = []
        #image_conditioning = []
        #seeds = []
        batch_n = 0
        batch_iter = get_batch_iter(trainer)
        while nb_samples < self.nb_samples:
            logger.info(f'sampling={nb_samples}')

            if batch_iter is not None:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    # restart the sequence
                    batch_iter = get_batch_iter()
                    batch = next(batch_iter)
                true_batches.append(self.unnorm_truth_fn(batch[self.input_name]).detach().cpu())
            batch = transfer_batch_to_device(batch, pl_module.device)

            # optional: specify the initial seeds
            # this is useful to understand the effect of 
            # the training over time
            """
            if self.deterministic_seed:
                if len(self.seeds) == 0:
                    x_T = torch.randn_like(batch[self.input_name])
                    seeds.append(x_T)
                else:
                    x_T = self.seeds[batch_n]
            else:
                x_T = torch.randn_like(batch[self.input_name])
            """

            encoding = pl_module.encode(batch, **self.encode_kwargs)
            image_decoded = pl_module.decode(encoding, **self.encode_kwargs)
            image_sampled.append(self.unnorm_output_fn(image_decoded).detach().cpu())

            batch_n += 1
            nb_samples += len(image_decoded)


        # record the first seed so we can restart from the same
        # random state next evaluation
        #if len(seeds) > 0 and len(self.seeds) == 0:
        #    self.seeds = seeds

        logger.info('Sampling done!')
        output_folder = os.path.join(trainer.options.data.root_current_experiment, self.sampling_folder)
        os.makedirs(output_folder, exist_ok=True)

        export_grid(image_sampled, os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_sampled.png'), nb_samples, save_as_numpy=self.save_numpy)
        
        #if len(image_conditioning) > 0:
        #    export_grid(image_conditioning, os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_conditioning.png'), nb_samples, save_as_numpy=self.save_numpy)
        export_grid(true_batches, os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_trues.png'), nb_samples, save_as_numpy=self.save_numpy)
