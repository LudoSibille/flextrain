from typing import Dict, Sequence, Optional, Callable, Tuple
import torch
from torch import nn
import numpy as np

from .utils import expand_dim_like
from ..types import TorchTensorNX, TorchTensorN


@torch.no_grad()
def forward_gaussian_diffusion(x0: TorchTensorNX, t: TorchTensorN, ᾱ: TorchTensorN) -> Tuple[TorchTensorNX, TorchTensorNX]:
    """
    Sample x0 at a given time step t

    Calculate q(xt | x0)

    Returns:
        the q(xt | x0) and Ɛ
    """
    sqrt_ᾱ = torch.sqrt(ᾱ[t])
    sqrt_one_minus_ᾱ = torch.sqrt(1 - ᾱ[t])
    Ɛ = torch.randn_like(x0)

    # enable broadcasting to replicate the missing dimensions 
    sqrt_ᾱ = expand_dim_like(sqrt_ᾱ, x0)
    sqrt_one_minus_ᾱ = expand_dim_like(sqrt_one_minus_ᾱ, x0)

    # equation (4) of the paper [1] 
    #  * mean = sqrt(ᾱ)
    #  * standard deviation = sqrt(1 - ᾱ)
    return sqrt_ᾱ * x0 + sqrt_one_minus_ᾱ * Ɛ, Ɛ


@torch.no_grad()
def sample_ddpm(
        α: TorchTensorN, 
        ᾱ: TorchTensorN, 
        σ: TorchTensorN, 
        device: torch.device, 
        model: nn.Module, 
        batch_shape: Sequence[int], 
        x_T: Optional[TorchTensorNX] = None, 
        model_extra_kwargs: Dict = {}) -> TorchTensorNX:
    """
    Sample an image from random noise

    This is the implementation of Algorithm 2 in [1].

    [1] Denoising Diffusion Probabilistic Models, J Ho et Al., https://arxiv.org/pdf/2006.11239.pdf
    """
    assert len(α) == len(ᾱ) == len(σ)
    model.eval()
    batch_size = batch_shape[0]
    x_t = torch.randn(batch_shape, device=device) if x_T is None else x_T.to(device)
    for t in reversed(range(1, len(α))):
        tt = torch.full([len(x_t)], fill_value=t, device=device, dtype=int)
        predicted_noise = model(x_t, tt, **model_extra_kwargs)

        # do NOT add noise for the final step
        noise = torch.randn_like(x_t) if t > 1 else 0
        x_t = 1 / torch.sqrt(α[t]) * (x_t - ((1 - α[t]) / (torch.sqrt(1 - ᾱ[t]))) * predicted_noise) + σ[t] * noise

    model.train()
    return x_t


@torch.no_grad()
def sample_ddpm_ilvr(
        α: TorchTensorN, 
        ᾱ: TorchTensorN, 
        σ: TorchTensorN, 
        device: torch.device, 
        model: nn.Module, 
        batch_shape: Sequence[int], 
        y: TorchTensorNX,
        lowpass_filter_fn: Callable[[TorchTensorNX], TorchTensorNX],
        x_T: Optional[TorchTensorNX] = None,
        stop_t: int = 0,
        model_extra_kwargs: Dict = {}) -> TorchTensorNX:
    """
    Sample an image from random noise such that the low pass filter(sampled) == low pass filter(y)

    This is the implementation of Algorithm 1 in [1].

    [1] ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models, J Choi et Al., https://arxiv.org/pdf/2108.02938.pdf
    """
    assert len(α) == len(ᾱ) == len(σ)
    model.eval()
    batch_size = batch_shape[0]
    
    x_t = torch.randn(batch_shape, device=device) if x_T is None else x_T.to(device)
    for t in reversed(range(1, len(α))):
        tt = torch.full([batch_size], fill_value=t, device=device, dtype=int)
        predicted_noise = model(x_t, tt, **model_extra_kwargs)

        # do NOT add noise for the final step
        noise = torch.randn_like(x_t) if t > 1 else 0
        x_t_prime = 1 / torch.sqrt(α[t]) * (x_t - ((1 - α[t]) / (torch.sqrt(1 - ᾱ[t]))) * predicted_noise) + σ[t] * noise

        if t > stop_t:
            y_t, _ = forward_gaussian_diffusion(y, tt, ᾱ)
            x_t = x_t_prime + lowpass_filter_fn(y_t) - lowpass_filter_fn(x_t_prime)
        else:
            x_t = x_t_prime

    model.train()
    return x_t


@torch.no_grad()
def sample_ddpm_kg(
        α: TorchTensorN, 
        ᾱ: TorchTensorN, 
        σ: TorchTensorN, 
        σ_d: TorchTensorN,
        device: torch.device, 
        model: nn.Module, 
        batch_shape: Sequence[int], 
        noisy: TorchTensorNX,
        x_T: Optional[TorchTensorNX] = None,
        stop_t: int = 0,
        model_extra_kwargs: Dict = {}) -> TorchTensorNX:
    """
    Unconditional image denoising

    This is the implementation of Algorithm 1 in [1].

    [1] PET image denoising based on denoising diffusion probabilistic models, K. Gong et al, https://arxiv.org/pdf/2209.06167.pdf
    """
    assert len(α) == len(ᾱ) == len(σ)
    if x_T is not None:
        assert len(noisy) == len(x_T)
    assert len(σ_d) == len(noisy)

    model.eval()
    batch_size = batch_shape[0]
        
    x_t = torch.randn(batch_shape, device=device) if x_T is None else x_T.to(device)
    for t in reversed(range(1, len(α))):
        tt = torch.full([batch_size], fill_value=t, device=device, dtype=int)
        predicted_noise = model(x_t, tt, **model_extra_kwargs)

        # do NOT add noise for the final step
        noise = torch.randn_like(x_t) if t > 1 else 0
        x_t_new = 1 / torch.sqrt(α[t]) * (x_t - ((1 - α[t]) / (torch.sqrt(1 - ᾱ[t]))) * predicted_noise) + σ[t] * noise

        if t > stop_t:                
            r = σ[t] ** 2 / σ_d ** 2
            r = expand_dim_like(r, x_t_new)
            
            # this works better than the original formulation
            noisy_t, _ = forward_gaussian_diffusion(noisy, t, ᾱ)
            x_t = x_t_new + (noisy_t - x_t) * r

            # original formulation
            #x_t = x_t_new + (noisy - x_t) * r
            #x_t = x_t_new + (noisy - x_t_new) * r

        else:
            x_t = x_t_new

    model.train()
    return x_t


@torch.no_grad()
def sample_ddim(
        α: TorchTensorN, 
        ᾱ: TorchTensorN, 
        σ: TorchTensorN, 
        device: torch.device, 
        model: nn.Module, 
        batch_shape: Sequence[int], 
        x_T: Optional[TorchTensorNX] = None, 
        nb_steps: int = 150,
        eta: float = 0.0,
        model_extra_kwargs: Dict = {}) -> TorchTensorNX:
    """
    Sample an image from random noise

    This is the implementation of equation 12 in [1].

    Args:
        eta: the random noise fraction injected at each step. eta = 0 is DDIM, no randomness
            in the sampling

    Note:
        This paper is using a different notation from DDPM paper. Eq (12) use `α` for `ᾱ`

    [1] Denoising Diffusion Implicit Models, Jiaming Song, Chenlin Meng and Stefano Ermon, https://arxiv.org/pdf/2010.02502.pdf
    """
    assert len(α) == len(ᾱ) == len(σ)
    model.eval()
    batch_size = batch_shape[0]

    x_t = torch.randn(batch_shape, device=device) if x_T is None else x_T.to(device)
    t_steps = np.linspace(len(α) - 1 - 1, 1, nb_steps, dtype=int)
    for i, t in enumerate(t_steps):
        tt = torch.full([batch_size], fill_value=t, device=device, dtype=int)
        predicted_noise = model(x_t, tt, **model_extra_kwargs)

        # do NOT add noise for the final step
        noise = torch.randn_like(x_t) if t > 1 else 0

        ᾱt1 = ᾱ[t_steps[i + 1]] if t > 1 else torch.tensor(1)
        vari = (1 - ᾱt1) / (1 - ᾱ[t]) * (1 - ᾱ[t] / ᾱt1)
        sig = vari.sqrt() * eta
        x_0_hat = (x_t - (1 - ᾱ[t]).sqrt() * predicted_noise) / ᾱ[t].sqrt()
        x_t = ᾱt1.sqrt() * x_0_hat + (1 - ᾱt1 - sig ** 2).sqrt() * predicted_noise + sig * noise

    model.train()
    return x_t