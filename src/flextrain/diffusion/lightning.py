from functools import partial
from typing import Any, Callable, Optional, Sequence, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..trainer.utils import postprocess_optimizer_scheduler_lightning
from ..types import Batch, TorchTensorNX
from .types import Model


class GaussianDiffusionLightning(L.LightningModule):
    def __init__(
        self,
        model: Model,
        input_name: str,
        input_conditioning_names: Optional[Union[str, Sequence[str]]] = None,
        optimizer_fn: Callable[[L.LightningModule], torch.optim.Optimizer] = partial(torch.optim.Adam, lr=5e-4),
        scheduler_steps_fn: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_name = input_name
        self.input_conditioning_names = input_conditioning_names
        self.optimizer_fn = optimizer_fn
        self.scheduler_steps_fn = scheduler_steps_fn

        if isinstance(self.input_conditioning_names, str):
            self.input_conditioning_names = [self.input_conditioning_names]

    def _step(self, batch: Batch) -> TorchTensorNX:
        i = batch[self.input_name]
        input_conditioning = {}
        if self.input_conditioning_names is not None:
            input_conditioning = {name: batch[name] for name in self.input_conditioning_names}

        losses = self.model.training_step(i, **input_conditioning)
        del losses['timesteps']  # not part of the loss

        # log all the losses
        loss_name_postfix = ''
        if self.training:
            loss_name_postfix = '_train'
        else:
            loss_name_postfix = '_valid'

        loss = 0
        for loss_name, loss_value in losses.items():
            with torch.no_grad():
                # MUST have `on_epoch` to have average over the whole epoch
                self.log(loss_name + loss_name_postfix, loss_value.mean(), prog_bar=True, on_step=True, on_epoch=True)

        loss = sum(losses.values())
        return loss

    def training_step(self, batch: Batch, _: Any) -> TorchTensorNX:
        if batch is None:
            return None

        return self._step(batch=batch).mean()

    def validation_step(self, batch: Batch, _: Any) -> Union[STEP_OUTPUT, None]:
        if batch is None:
            return None

        with torch.no_grad():
            return self._step(batch=batch).mean()

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())
        if self.scheduler_steps_fn is None:
            return optimizer

        scheduler = self.scheduler_steps_fn(optimizer, self)
        return postprocess_optimizer_scheduler_lightning(optimizer, scheduler)

    def sample(self, **kwargs) -> TorchTensorNX:
        if self.input_conditioning_names is not None:
            for name in self.input_conditioning_names:
                assert kwargs.get(name) is not None, f'missing conditioning input={name}'
        return self.model.sample(**kwargs)
