from typing import Any, Dict, Optional, Sequence, Tuple, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from ..contrastive.lightning import process_loss_outputs
from ..losses import LossOutput
from ..trainer.utils import len_batch
from ..types import Batch, TorchTensorNX
from .gan_cycle import LambdaLR


def get_default_optim_params() -> Dict:
    return {
        'lr': 2e-4,
        'betas': (0.5, 0.999),
    }


class GanDC(L.LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        image_name: str,
        latent_size: int = 100,
        generator_conditional_feature_names: Sequence[str] = (),
        discriminator_conditional_feature_names: Sequence[str] = (),
        optimizer_params: Dict = get_default_optim_params(),
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.image_name = image_name
        self.latent_size = latent_size
        self.generator_conditional_feature_names = generator_conditional_feature_names
        self.discriminator_conditional_feature_names = discriminator_conditional_feature_names
        self.optimizer_params = optimizer_params
        self.label_smoothing = label_smoothing

        # manual optimization steps
        self.automatic_optimization = False

        self.adversarial_loss = torch.nn.BCELoss()

    def _step(self, batch: Batch) -> TorchTensorNX:
        g_opt, d_opt = self.optimizers()
        lr_scheduler_g, lr_scheduler_d = self.lr_schedulers()

        batch_size = len_batch(batch)
        X = batch[self.image_name]
        real_label = torch.full((batch_size, 1), fill_value=1.0 - self.label_smoothing, device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_conditional = {n: batch[n] for n in self.generator_conditional_feature_names}
        d_conditional = {n: batch[n] for n in self.discriminator_conditional_feature_names}

        g_X = self.sample_g(batch_size, **g_conditional)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.discriminator(X, **d_conditional)
        errD_real = self.adversarial_loss(d_x, real_label)

        d_g = self.discriminator(g_X.detach(), **d_conditional)
        errD_fake = self.adversarial_loss(d_g, fake_label)

        errD = errD_real + errD_fake

        if self.training:
            d_opt.zero_grad()
            self.manual_backward(errD)
            d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.discriminator(g_X, **d_conditional)
        errG = self.adversarial_loss(d_z, real_label)

        if self.training:
            g_opt.zero_grad()
            self.manual_backward(errG)
            g_opt.step()

        # Update learning rates (only in training mode)
        if self.training and self.trainer.is_last_batch:
            if lr_scheduler_g is not None:
                lr_scheduler_g.step()
            if lr_scheduler_d is not None:
                lr_scheduler_d.step()

        loss_outputs = LossOutput(
            metrics={
                'accuracy_d_real': ((d_x >= 0.5).sum().float()) / len(d_x),
                'accuracy_d_fake': ((d_g <= 0.5).sum().float()) / len(d_x),
            },
            losses={
                'generator': errG.detach(),
                'discriminator_real': errD_real.detach(),
                'discriminator_fake': errD_fake.detach(),
            },
        )

        return process_loss_outputs(self, loss_outputs, batch_size=batch_size)

    def training_step(self, batch: Batch, _: Any) -> TorchTensorNX:
        return self._step(batch=batch)

    def validation_step(self, batch: Batch, _: Any) -> Union[STEP_OUTPUT, None]:
        with torch.no_grad():
            return self._step(batch=batch)

    def forward(self, batch: Batch, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = len_batch(batch)
        generator_kwargs = {n: batch[n] for n in self.generator_conditional_feature_names}
        return self.sample_g(batch_size=batch_size, z=z, **generator_kwargs)

    def sample_z(self, batch_size: int) -> TorchTensorNX:
        z = torch.randn(batch_size, self.latent_size, dtype=torch.float32, device=self.device)
        return z

    def sample_g(self, batch_size: int, z: Optional[torch.Tensor] = None, **generator_kwargs: Any) -> TorchTensorNX:
        if z is None:
            z = self.sample_z(batch_size)

        return self.generator(z, **generator_kwargs)

    def configure_optimizers(self) -> Tuple[Sequence[torch.optim.Optimizer], Sequence[torch.optim.lr_scheduler.LRScheduler]]:
        n_epochs = self.trainer.options.training.nb_epochs
        epoch = self.trainer.current_epoch
        decay_epoch = int(n_epochs * 0.5)

        opt_g = torch.optim.Adam(self.generator.parameters(), **self.optimizer_params)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), **self.optimizer_params)

        # Learning rate update schedulers
        lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
        lr_scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

        return [opt_g, opt_d], [lr_scheduler_g, lr_scheduler_d]
