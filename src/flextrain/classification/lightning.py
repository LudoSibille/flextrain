from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from ..contrastive.lightning import process_loss_outputs
from ..losses import loss_function_type
from ..trainer.optimization import scheduler_steps_fn
from ..trainer.utils import len_batch, postprocess_optimizer_scheduler_lightning
from ..types import Batch, TorchTensorNX


class ClassifierLightning(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: loss_function_type,
        optimizer_fn: Callable[[L.LightningModule], torch.optim.Optimizer] = partial(
            torch.optim.SGD, lr=0.1, nesterov=True, weight_decay=5e-5, momentum=0.99
        ),
        scheduler_steps_fn: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        ] = scheduler_steps_fn,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_fn = optimizer_fn
        self.scheduler_steps_fn = scheduler_steps_fn
        self.loss_fn = loss_fn

    def _step(self, batch: Batch) -> TorchTensorNX:
        output = self.model(batch)
        loss_outputs = self.loss_fn(batch=batch, model_output=output)
        return process_loss_outputs(self, loss_outputs, batch_size=len_batch(batch))

    def training_step(self, batch: Batch, _: Any) -> TorchTensorNX:
        return self._step(batch=batch)

    def validation_step(self, batch: Batch, _: Any) -> Union[STEP_OUTPUT, None]:
        with torch.no_grad():
            return self._step(batch=batch)

    def forward(self, batch: Batch) -> Any:
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())
        if self.scheduler_steps_fn is None:
            return optimizer

        scheduler = self.scheduler_steps_fn(optimizer, self)
        return postprocess_optimizer_scheduler_lightning(optimizer, scheduler)
