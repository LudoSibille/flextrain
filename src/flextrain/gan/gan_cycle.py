import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from ..contrastive.lightning import process_loss_outputs
from ..losses import LossOutput
from ..trainer.utils import len_batch
from ..types import Batch, TorchTensorNX


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ReplayBuffer:
    def __init__(self, max_size: int = 50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data: List[torch.Tensor] = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def get_default_optim_params() -> Dict:
    return {
        'lr': 2e-4,
        'betas': (0.5, 0.999),
    }


class GanCycle(L.LightningModule):
    """
    Implement a basic Cycle GAN.

    TODO:
    - implement the conditioning
    """

    def __init__(
        self,
        generator_AB: nn.Module,
        discriminator_A: nn.Module,
        image_name_A: str,
        generator_BA: nn.Module,
        discriminator_B: nn.Module,
        image_name_B: str,
        generator_conditional_feature_names_A: Sequence[str] = (),
        discriminator_conditional_feature_names_A: Sequence[str] = (),
        generator_conditional_feature_names_B: Sequence[str] = (),
        discriminator_conditional_feature_names_B: Sequence[str] = (),
        lambda_cyc: float = 10.0,
        lambda_id: float = 5.0,
        optimizer_params_generator: Dict = get_default_optim_params(),
        optimizer_params_discriminator: Dict = get_default_optim_params(),
        forward_generate_A: bool = False,
    ) -> None:
        super().__init__()
        self.generator_AB = generator_AB
        self.discriminator_A = discriminator_A
        self.generator_BA = generator_BA
        self.discriminator_B = discriminator_B
        self.image_name_A = image_name_A
        self.image_name_B = image_name_B
        self.generator_conditional_feature_names_A = generator_conditional_feature_names_A
        self.discriminator_conditional_feature_names_A = discriminator_conditional_feature_names_A
        self.generator_conditional_feature_names_B = generator_conditional_feature_names_B
        self.discriminator_conditional_feature_names_B = discriminator_conditional_feature_names_B
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id
        self.optimizer_params_generator = optimizer_params_generator
        self.optimizer_params_discriminator = optimizer_params_discriminator

        # manual optimization steps: we will handle the
        # optimization, not torch.Lightning
        self.automatic_optimization = False

        self.criterion_gan = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.forward_generate_A = forward_generate_A

    def _step(self, batch: Batch) -> TorchTensorNX:
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = self.lr_schedulers()
        batch_size = len_batch(batch)
        real_A = batch[self.image_name_A]
        real_B = batch[self.image_name_B]

        # real_label = torch.ones((batch_size, 1), device=self.device, requires_grad=False)
        # fake_label = torch.zeros((batch_size, 1), device=self.device, requires_grad=False)

        real_label = torch.ones((1,), device=self.device, requires_grad=False)
        fake_label = torch.zeros((1,), device=self.device, requires_grad=False)

        ######################
        #  Train Generators  #
        ######################
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = self.criterion_identity(self.generator_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.generator_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.generator_AB(real_A)
        loss_GAN_AB = self.criterion_gan(self.discriminator_B(fake_B), real_label)
        fake_A = self.generator_BA(real_B)
        loss_GAN_BA = self.criterion_gan(self.discriminator_A(fake_A), real_label)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.generator_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.generator_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity

        # loss_G.backward()
        if self.training:
            self.manual_backward(loss_G)
            optimizer_G.step()

        ###########################
        #  Train Discriminator A  #
        ###########################

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = self.criterion_gan(self.discriminator_A(real_A), real_label)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
        loss_fake = self.criterion_gan(self.discriminator_A(fake_A_.detach()), fake_label)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        if self.training:
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()

        ###########################
        #  Train Discriminator B  #
        ###########################

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = self.criterion_gan(self.discriminator_B(real_B), real_label)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
        loss_fake = self.criterion_gan(self.discriminator_B(fake_B_.detach()), fake_label)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        if self.training:
            self.manual_backward(loss_D_B)
            optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # Update learning rates (only in training mode)
        if self.training and self.trainer.is_last_batch:
            if lr_scheduler_G is not None:
                lr_scheduler_G.step()
            if lr_scheduler_D_A is not None:
                lr_scheduler_D_A.step()
            if lr_scheduler_D_B is not None:
                lr_scheduler_D_B.step()

        loss_outputs = LossOutput(
            metrics={
                #'accuracy_d_real': ((d_x >= 0.5).sum().float()) / len(d_x),
                #'accuracy_d_fake': ((d_g <= 0.5).sum().float()) / len(d_x),
            },
            losses={
                'discriminator': loss_D.detach(),
                'generator': loss_G.detach(),
                'GAN': loss_GAN.detach(),
                'cycle': loss_cycle.detach(),
                'identity': loss_identity.detach(),
            },
        )

        return process_loss_outputs(self, loss_outputs, batch_size=batch_size)

    def training_step(self, batch: Batch, _: Any) -> TorchTensorNX:
        return self._step(batch=batch)

    def validation_step(self, batch: Batch, _: Any) -> Union[STEP_OUTPUT, None]:
        with torch.no_grad():
            return self._step(batch=batch)

    def forward(self, batch: Batch) -> torch.Tensor:
        batch_size = len_batch(batch)
        z = self.sample_z(batch_size)
        if self.forward_generate_A:
            generator_kwargs = {n: batch[n] for n in self.generator_conditional_feature_names_A}
            return self.sample_A(batch_size=batch_size, z=z, B=batch[self.image_name_B], **generator_kwargs)
        else:
            generator_kwargs = {n: batch[n] for n in self.generator_conditional_feature_names_B}
            return self.sample_B(batch_size=batch_size, z=z, A=batch[self.image_name_A], **generator_kwargs)

    def sample_z(self, batch_size: int) -> Optional[torch.Tensor]:
        # no seed: the original CycleGAN doesn't use a random seed
        return None

    def sample_B(
        self, batch_size: int, A: torch.Tensor, z: Optional[torch.Tensor] = None, **generator_kwargs: Any
    ) -> TorchTensorNX:
        return self.generator_AB(A, **generator_kwargs)

    def sample_A(
        self, batch_size: int, B: torch.Tensor, z: Optional[torch.Tensor] = None, **generator_kwargs: Any
    ) -> TorchTensorNX:
        return self.generator_BA(B, **generator_kwargs)

    def configure_optimizers(self) -> Tuple[Sequence[torch.optim.Optimizer], Sequence[torch.optim.lr_scheduler.LRScheduler]]:
        n_epochs = self.trainer.options.training.nb_epochs
        epoch = self.trainer.current_epoch
        decay_epoch = int(n_epochs * 0.5)

        optimizer_G = torch.optim.Adam(
            itertools.chain(self.generator_AB.parameters(), self.generator_BA.parameters()),
            **self.optimizer_params_generator,
        )
        optimizer_D_A = torch.optim.Adam(self.discriminator_A.parameters(), **self.optimizer_params_discriminator)
        optimizer_D_B = torch.optim.Adam(self.discriminator_B.parameters(), **self.optimizer_params_discriminator)

        # Learning rate update schedulers
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        )
        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]
