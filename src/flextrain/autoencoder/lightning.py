from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from ..contrastive.lightning import process_loss_outputs
from ..losses import loss_function_type
from ..trainer.optimization import scheduler_steps_fn
from ..trainer.utils import len_batch, postprocess_optimizer_scheduler_lightning
from ..types import Batch, TorchTensorNX
from .ae import AutoEncoderType


class AutoencoderLightning(L.LightningModule):
    def __init__(
        self,
        ae: AutoEncoderType,
        loss_fn: loss_function_type,
        optimizer_fn: Callable[[L.LightningModule], torch.optim.Optimizer] = partial(
            torch.optim.SGD, lr=0.1, nesterov=True, weight_decay=5e-5, momentum=0.99
        ),
        scheduler_steps_fn: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        ] = scheduler_steps_fn,
    ) -> None:
        super().__init__()
        self.ae = ae
        self.loss_fn = loss_fn

        self.optimizer_fn = optimizer_fn
        self.scheduler_steps_fn = scheduler_steps_fn

    def _step(self, batch: Batch) -> TorchTensorNX:
        encoding = self.ae.encode(batch)
        if isinstance(encoding, tuple):
            output = self.ae.decode(*encoding)
        else:
            output = self.ae.decode(encoding)
        loss_outputs = self.loss_fn(batch=batch, encoding=encoding, model_output=output)
        return process_loss_outputs(self, loss_outputs, batch_size=len_batch(batch))

    def training_step(self, batch: Batch, _: Any) -> TorchTensorNX:
        return self._step(batch=batch)

    def validation_step(self, batch: Batch, _: Any) -> Union[STEP_OUTPUT, None]:
        with torch.no_grad():
            return self._step(batch=batch)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(images)

    def encode(self, *args, **kwargs):
        return self.ae.encode(*args, **kwargs)

    def decode(self, encoding, *args, **kwargs):
        if isinstance(encoding, tuple):
            return self.ae.decode(*encoding, *args, **kwargs)

        return self.ae.decode(encoding, *args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())
        if self.scheduler_steps_fn is None:
            return optimizer

        scheduler = self.scheduler_steps_fn(optimizer, self)
        return postprocess_optimizer_scheduler_lightning(optimizer, scheduler)
