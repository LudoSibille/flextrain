import math
from typing import Optional, Tuple
import torch
from dataclasses import dataclass

@dataclass
class NoisingSchedule:
    α: torch.Tensor
    ᾱ: torch.Tensor
    σ: torch.Tensor


def get_ᾱ_from_β(β: torch.Tensor):
    """
    Calculate ᾱ from the β schedule
    """
    α = 1 - β
    ᾱ = torch.cumprod(α, dim=0)
    assert (ᾱ > 0).all(), 'to avoid singularities, ALL ᾱ must be strictly greater than 0'
    return ᾱ


def noising_schedule_from_β(β: torch.Tensor) -> NoisingSchedule:
    return NoisingSchedule(
        α=1 - β,
        ᾱ=get_ᾱ_from_β(β),
        σ=β.sqrt()
    )


def enforce_zero_terminal_snr(betas: torch.Tensor, eps: float=1e-6):
    """
    diffusion models should use noise schedules with zero terminal SNR and
    should be sampled starting from the last timestep in order to
    ensure the training behavior is aligned with inference

    [1] https://arxiv.org/pdf/2305.08891.pdf

    Returns:
        betas corrected tyo have zero terminal SNR
    """

    # use higher precisions 
    dtype_orig = betas.dtype
    betas = betas.type(torch.float64)

    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / (alphas_bar[:-1])
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas.type(dtype_orig)
    betas = betas.type(dtype_orig)

    betas[-1] -= eps  # avoid last betas to be exactly 1 to avoid singularities
    assert (betas > 0).all(), 'avoid singularities. Something is wrong with the parameters. Precision? Too large β_end?'
    assert (betas < 1).all(), 'avoid singularities'
    return betas


def linear(β_start: float = 1e-4, β_end: float = 0.02, steps: int = 1000) -> NoisingSchedule:
    """
    Linearly sample T within a given interval

    Returns:
        β
    """
    # the schedule should be as independent as possible from the
    # number of time steps. Else, our parameters are dependent on
    # steps, making difficult to compare different hyper-parameters
    # so scale the coefficients accordingly
    scale = 1000 / steps

    # float64 for increased precision. This is needed for the final time steps
    #  to calculate accurately `alphas.cumprod`
    β = torch.linspace(scale * β_start, scale * β_end, steps)
    return noising_schedule_from_β(β)


def linear_zero_snr(*args, **kwargs):
    """
    Corrected linear β to have zero SNR at timestep T
    """
    return enforce_zero_terminal_snr(linear(*args, **kwargs))


def cosine_beta_schedule(steps: int = 1000) -> NoisingSchedule:
    """
    Cosine schedule inspired from https://arxiv.org/abs/2102.09672
    """
    abar = lambda t: (t / steps * math.pi / 2).cos() ** 2

    x = torch.linspace(0, steps - 1, steps)
    ᾱ = abar(x)
    α = ᾱ / abar(x - 1)
    return NoisingSchedule(α=α, ᾱ=ᾱ, σ=(1 - α).sqrt())


def sigmoid_beta_schedule(β_start: float = 1e-4, β_end: float = 0.02, sig_range: float = 6, steps: int = 1000) -> NoisingSchedule:
    """
    Sigmoid beta noise schedule function.

    Args:
        steps: number of timesteps
        β_start: start of beta range
        β_end: end of beta range
        sig_range: pos/neg range of sigmoid input

    Returns:
        betas: beta schedule tensor
    """
    scale = 1000 / steps
    x = torch.linspace(-sig_range, sig_range, steps)
    β = torch.sigmoid(x) * (scale * β_end - scale * β_start) + scale * β_start
    return noising_schedule_from_β(β)


def scaled_linear_beta(β_start: float = 1e-4, β_end: float = 2e-2, power: int = 2, steps: int = 1000) -> NoisingSchedule:
    """
    Scaled linear beta noise schedule function.

    Args:
        steps: number of timesteps
        β_start: start of beta range, default 1e-4
        β_end: end of beta range, default 2e-2

    Returns:
        betas: beta schedule tensor
    """
    β = torch.linspace(β_start**(1.0 / power), β_end**(1.0 / power), steps, dtype=torch.float32) ** power
    return noising_schedule_from_β(β)
