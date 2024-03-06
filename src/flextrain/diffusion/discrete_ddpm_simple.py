from functools import partial
from typing import Callable, Dict, Sequence, Tuple, Optional
import torch

from .discrete_beta_schedulers import linear, NoisingSchedule
from .discrete_sampling import sample_ddpm, forward_gaussian_diffusion
from .discrete_scheduler_time import rand_uniform_steps
from .utils import expand_dim_like
from ..types import TorchTensorN, TorchTensorNX
from .types import Model


# standard formulation is using L2 loss (MSE)
loss_mse = partial(torch.nn.functional.mse_loss, reduction='none')

# https://arxiv.org/pdf/2111.05826.pdf found L1 decreased
# sample diversity and maybe hallucinations too
# which could be useful for some applications
loss_l1 = partial(torch.nn.functional.l1_loss, reduction='none')


class SimpleGaussianDiffusion(Model):
    """
    The Simplest implementation of a Denoising diffusion probabilitic model as described in [1].

    This is using the simplified training objective (Algorithm 1 and 2 of the paper).

    The Ɛ-error is fitted by the model.
    
    [1] Denoising Diffusion Probabilistic Models, J Ho et Al, https://arxiv.org/pdf/2006.11239.pdf
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            noise_scheduler_fn: Callable[[int], NoisingSchedule] = linear,
            timestep_training_fn: Callable[[int, int], int] = rand_uniform_steps,
            sample_fn = sample_ddpm,
            loss_fn = loss_mse):
        """
        """
        super().__init__()
        sched = noise_scheduler_fn()
        assert len(sched.ᾱ.shape) == 1
        assert sched.ᾱ.shape == sched.σ.shape
        assert sched.ᾱ.shape == sched.α.shape
        self.noise_steps = len(sched.ᾱ)

        self.model = model
        self.loss_fn = loss_fn
        self.sample_fn = sample_fn
        self.timestep_training_fn = timestep_training_fn

        # make sure the tensors are moved to the correct device
        # by registering them as buffers
        self.register_buffer('α', sched.α)
        self.register_buffer('ᾱ', sched.ᾱ)
        self.register_buffer('σ', sched.σ)


    def sample_timesteps(self, number_of_samples: int, device: torch.device = None) -> torch.Tensor:
        # We should only sample t >= 1 as t=0 we have x0 = input
        #
        return self.timestep_training_fn(number_of_samples, self.noise_steps, device=device, dtype=torch.int32)


    def forward_diffusion(self, x0: TorchTensorNX, t: TorchTensorN) -> Tuple[TorchTensorNX, TorchTensorNX]:
        """
        Sample x0 at a given time step t

        Calculate q(xt | x0)

        Returns:
            the q(xt | x0) and Ɛ
        """
        return forward_gaussian_diffusion(x0=x0, t=t, ᾱ=self.ᾱ)

    def sample(self, batch_shape: Sequence[int], x_T: Optional[TorchTensorNX] = None, **model_extra_kwargs) -> TorchTensorNX:
        return self.sample_fn(
            α=self.α,
            ᾱ=self.ᾱ,
            σ=self.σ,
            device=self.α.device,
            model=self.model,
            batch_shape=batch_shape,
            x_T=x_T,
            model_extra_kwargs=model_extra_kwargs
        )

    def training_step(self, x0: TorchTensorNX, **model_extra_kwargs) -> Dict[str, TorchTensorNX]:
        """
        Calculate the loss for one training step
        """
        t = self.sample_timesteps(x0.shape[0], device=x0.device)

        x_t, Ɛ = self.forward_diffusion(x0, t)   
        predicted_Ɛ = self.model(x_t, t, **model_extra_kwargs)
        loss = self.loss_fn(Ɛ, predicted_Ɛ)
        assert len(loss) == len(x0), 'each sample must have an associated loss'
        
        # mean per-pixel loss and keep a single loss per sample
        loss = loss.view(loss.shape[0], -1).mean(dim=1)
        return {'loss': loss, 'timesteps': t}
