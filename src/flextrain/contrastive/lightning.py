from functools import partial
from typing import Any, Callable, Optional, Sequence, Union
from warnings import warn

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from ..losses import LossOutput
from ..trainer.optimization import scheduler_steps_fn
from ..trainer.utils import len_batch, postprocess_optimizer_scheduler_lightning
from ..types import Batch, TorchTensorNX


def process_loss_outputs(model: L.LightningModule, loss_outputs: LossOutput, batch_size: int) -> torch.Tensor:
    # log all the losses
    loss_name_postfix = ''
    if model.training:
        loss_name_postfix = '_train'
    else:
        loss_name_postfix = '_valid'

    kwargs = {
        'on_epoch': True,
        'on_step': True,
        # 'batch_size': batch_size
    }

    def to_device(v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(model.device)
        return torch.tensor(v, device=model.device)

    loss: torch.Tensor = 0
    for loss_name, loss_by_sample in loss_outputs.losses.items():
        if len(loss_by_sample.shape) == 0 or len(loss_by_sample) != batch_size:
            warn(f'expecting a loss per sample! Got shape={loss_by_sample.shape} instead of={batch_size}')
        # assert len(loss_by_sample.shape) >= 1 and len(loss_by_sample) == batch_size
        current_loss = loss_by_sample.mean()
        loss += current_loss
        model.log(
            f'loss_{loss_name}{loss_name_postfix}',
            # current_loss,
            to_device(current_loss),
            prog_bar=len(loss_outputs.losses) == 1,
            batch_size=batch_size,
            sync_dist=True,
            **kwargs,
        )

    if len(loss_outputs.losses) > 1:
        model.log(
            f'loss{loss_name_postfix}',
            # loss,
            to_device(loss),
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
            **kwargs,
        )

    for metric_name, metric_by_sample in loss_outputs.metrics.items():
        metric = metric_by_sample.mean()
        model.log(
            f'metric_{metric_name}{loss_name_postfix}',
            # metric,
            to_device(metric),
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
            **kwargs,
        )

    return loss


class MultiBatchLightning(L.LightningModule):
    """
    Several batches were merged into a single batch (e.g., learning ranking, triplet losses, contrastive)
    """

    def __init__(
        self,
        model: nn.Module,
        split_batch_fn: Callable[[Batch], Sequence[Batch]],
        loss_fn: Callable[[Sequence[TorchTensorNX], Sequence[Batch]], TorchTensorNX],
        optimizer_fn: Callable[[L.LightningModule], torch.optim.Optimizer] = partial(torch.optim.Adam, lr=1e-3),
        scheduler_steps_fn: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        ] = scheduler_steps_fn,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_fn = optimizer_fn
        self.scheduler_steps_fn = scheduler_steps_fn
        self.loss_fn = loss_fn
        self.split_batch_fn = split_batch_fn

    def _step(self, batch: Batch) -> TorchTensorNX:
        batches = self.split_batch_fn(batch)
        outputs = []
        for b in batches:
            o = self.model(b)
            outputs.append(o)

        loss_outputs = self.loss_fn(batches, outputs)
        batch_size = len_batch(batch)
        return process_loss_outputs(self, loss_outputs, batch_size=batch_size)

    def training_step(self, batch: Batch, _: Any) -> TorchTensorNX:
        return self._step(batch=batch).mean()

    def validation_step(self, batch: Batch, _: Any) -> Union[STEP_OUTPUT, None]:
        with torch.no_grad():
            return self._step(batch=batch).mean()

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters())
        if self.scheduler_steps_fn is None:
            return optimizer

        scheduler = self.scheduler_steps_fn(optimizer, self)
        return postprocess_optimizer_scheduler_lightning(optimizer, scheduler)

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.model(batch)
