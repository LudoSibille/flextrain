import logging
import os
from typing import Callable, Iterator, Optional, Sequence

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only

from ..trainer.utils import len_batch, transfer_batch_to_device
from .autoencoder import export_grid, get_batch_iter

logger = logging.getLogger(__name__)


def unnorm_m1p1_255_fn(images: torch.Tensor) -> torch.Tensor:
    """
    Transform range [-1...1] to [0..255]
    """
    images = (((images + 1) / 2.0).clamp(0, 1) * 255).type(torch.uint8)
    return images


class CallbackGanRecon(L.Callback):
    def __init__(
        self,
        input_name: str,
        nb_samples: int = 32,
        sampling_folder: str = 'recon_gan',
        unnorm_truth_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = unnorm_m1p1_255_fn,
        unnorm_output_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = unnorm_m1p1_255_fn,
        unnorm_cond_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = unnorm_m1p1_255_fn,
        generator_conditional_feature_names: Sequence[str] = (),
        deterministic_seed: bool = True,
        save_numpy: bool = False,
        model_sample_z_fn: Optional[Callable[[int], torch.Tensor]] = None,
        model_sample_g_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        conditioning_image_names: Sequence[str] = (),
    ) -> None:

        super().__init__()
        self.nb_samples = nb_samples
        self.input_name = input_name

        self.sampling_folder = sampling_folder
        self.unnorm_truth_fn = unnorm_truth_fn if unnorm_truth_fn is not None else lambda x: x
        self.unnorm_output_fn = unnorm_output_fn if unnorm_output_fn is not None else lambda x: x
        self.deterministic_seed = deterministic_seed

        self.generator_conditional_feature_names = generator_conditional_feature_names
        self.save_numpy = save_numpy

        self.seeds = None
        self.conditionings = None
        self.true_batches = None

        self.model_sample_z_fn = model_sample_z_fn
        self.model_sample_g_fn = model_sample_g_fn

        self.unnorm_cond_fn = unnorm_cond_fn
        self.conditioning_image_names = conditioning_image_names

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger.info('Sampling started!')
        nb_samples = 0

        true_batches = []
        image_sampled = []
        conditionings = []
        seeds = []
        batch_n = 0
        batch_iter = get_batch_iter(trainer)
        while nb_samples < self.nb_samples:
            logger.info(f'sampling={nb_samples}')

            assert batch_iter is not None
            if self.seeds is None:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    # restart the sequence
                    batch_iter = get_batch_iter()
                    batch = next(batch_iter)
                batch_size = len_batch(batch)
                batch = transfer_batch_to_device(batch, pl_module.device)

                true_batch = self.unnorm_truth_fn(batch[self.input_name]).detach().cpu()

                # specific seed (e.g., CycleGAN -> No seed)
                if self.model_sample_z_fn is None:
                    sample_z_fn = pl_module.sample_z
                else:
                    sample_z_fn = self.model_sample_z_fn
                seed = sample_z_fn(batch_size)
                g_conditional = {n: batch[n] for n in self.generator_conditional_feature_names}
            else:
                # retrieve the fixed conditioning and seed
                # this is useful to understand the effect of
                # the training over time
                seed = self.seeds[batch_n]
                g_conditional = self.conditionings[batch_n]
                true_batch = self.true_batches[batch_n]
                batch_size = len(true_batch)

            seeds.append(seed)
            conditionings.append(g_conditional)
            true_batches.append(true_batch)

            if self.model_sample_g_fn is None:
                sample_g_fn = pl_module.sample_g
            else:
                sample_g_fn = self.model_sample_g_fn
            image_decoded = sample_g_fn(batch_size, z=seed, **g_conditional)
            image_sampled.append(self.unnorm_output_fn(image_decoded).detach().cpu())

            batch_n += 1
            nb_samples += len(image_decoded)

        assert len(seeds) == len(true_batches)
        assert len(seeds) == len(image_sampled)
        if self.deterministic_seed and self.seeds is None and len(seeds) > 0:
            # record the first seed so we can restart from the same
            # random state next evaluation
            self.seeds = seeds
            self.conditionings = conditionings
            self.true_batches = true_batches

        logger.info('Sampling done!')
        output_folder = os.path.join(trainer.options.data.root_current_experiment, self.sampling_folder)
        os.makedirs(output_folder, exist_ok=True)

        export_grid(
            image_sampled,
            os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_sampled.png'),
            nb_samples,
            save_as_numpy=self.save_numpy,
        )

        if len(conditionings) > 0 and len(conditionings[0]) > 0:
            # only consider image inputs
            conditioning_keys = set(conditionings[0].keys()).intersection(self.conditioning_image_names)
            for key in conditioning_keys:
                values = [self.unnorm_cond_fn(c[key]).cpu() for c in conditionings]
                export_grid(
                    values,
                    os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_{key}.png'),
                    nb_samples,
                    save_as_numpy=self.save_numpy,
                )

        export_grid(
            true_batches,
            os.path.join(output_folder, f'e_{trainer.current_epoch}_step_{trainer.global_step}_trues.png'),
            nb_samples,
            save_as_numpy=self.save_numpy,
        )
