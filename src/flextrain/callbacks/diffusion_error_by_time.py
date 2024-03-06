from collections import defaultdict
from typing import Any, Optional, Union, Sequence
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
import time
import logging
import torch
import numpy as np
from ..trainer.utils import to_value, transfer_batch_to_device
import matplotlib.pyplot as plt
import os


logger = logging.getLogger(__name__)


class CallbackDiffusionErrorByTime(L.Callback):
    """
    Collect the training error of the revert diffusion along the time dimension
    """
    def __init__(
            self,
            sample_kwargs: Any,
            input_name: Optional[str] = None,
            nb_samples: int = 50000,
            max_timestep: int = 1000,
            sampling_folder: str = 'timesteps', 
            input_conditioning_names: Optional[Union[str, Sequence[str]]] = None, 
            ) -> None:
        super().__init__()
        self.nb_samples = nb_samples
        self.input_name = input_name
        self.sample_kwargs = sample_kwargs
        self.max_timestep = max_timestep
        self.sampling_folder = sampling_folder

        self.input_conditioning_names = input_conditioning_names
        if input_conditioning_names is None:
            self.input_conditioning_names = ()
        elif isinstance(self.input_conditioning_names, str):
            self.input_conditioning_names = [self.input_conditioning_names]
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        logger.info('Sampling started!')
        nb_samples = 0

        def get_batch_iter():
            if trainer.val_dataloaders is not None:
                batch_iter = iter(trainer.val_dataloaders)
            else:
                batch_iter = iter(trainer.train_dataloader)
            assert batch_iter is not None, 'no dataloader!'
            return batch_iter

        batch_iter = get_batch_iter()
        batch_n = 0
        errors_sum = np.zeros(self.max_timestep, dtype=np.float64)
        time_counts = np.zeros(self.max_timestep, dtype=int)
        while nb_samples < self.nb_samples:
            logger.info(f'sampling={nb_samples}')

            if batch_iter is not None:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    # restart the sequence
                    batch_iter = get_batch_iter()
                    batch = next(batch_iter)
            batch = transfer_batch_to_device(batch, pl_module.device)
            batch_n += 1

            conditioning = {}
            for name in self.input_conditioning_names:
                value = batch.get(name)
                if isinstance(value, torch.Tensor):
                    value = value.to(pl_module.device)
                assert value is not None, f'conditioning={name}, not found!'
                conditioning[name] = value


            x0 = batch[self.input_name]
            nb_samples += len(x0)
            with torch.no_grad():
                metadata = pl_module.model.training_step(x0=x0, **conditioning)
            timesteps = to_value(metadata['timesteps'])
            losses = to_value(metadata['loss'])
            
            # BEWARE `time_counts[timesteps] += 1` is incorrect for modifying 
            # the same index several times! only one update would be done (np.max(timesteps) == 1)
            for l, t in zip(losses, timesteps):
                time_counts[t] += 1
                errors_sum[t] += l
        
        output_folder = os.path.join(trainer.options.data.root_current_experiment, self.sampling_folder)
        os.makedirs(output_folder, exist_ok=True)
        assert time_counts.sum() >= self.nb_samples, 'problem!'
        errors_mean = errors_sum / (time_counts + 1)
        
        plt.close()
        plt.plot(np.arange(1000), errors_mean)
        plt.savefig(os.path.join(output_folder, f'plot_{trainer.current_epoch}.png'))
        plt.close()

        logger.info(f'Done! output_folder={output_folder}')