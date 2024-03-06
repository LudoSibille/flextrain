from functools import partial
import math
from typing import Callable, Dict, Sequence, Optional
import torch
from ..types import TorchTensorNX
from .types import Model
from .discrete_ddpm_simple import loss_mse
from .utils import expand_dim_like

"""

#
# parameter change: MODEL first
#

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta: return sigma_to, 0.
    var_to,var_from = sigma_to**2,sigma_from**2
    sigma_up = min(sigma_to, eta * (var_to * (var_from-var_to)/var_from)**0.5)
    return (var_to-sigma_up**2)**0.5, sigma_up


@torch.no_grad()
def sample_euler_ancestral(x, sigs, i, model, eta=1.):
    sig,sig2 = sigs[i],sigs[i+1]
    denoised = denoise(model, x, sig)
    sigma_down,sigma_up = get_ancestral_step(sig, sig2, eta=eta)
    x = x + (x-denoised)/sig*(sigma_down-sig)
    return x + torch.randn_like(x)*sigma_up




@torch.no_grad()
def sample_heun(x, sigs, i, model, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    sig,sig2 = sigs[i],sigs[i+1]
    n = len(sigs)
    gamma = min(s_churn/(n-1), 2**0.5-1) if s_tmin<=sig<=s_tmax else 0.
    eps = torch.randn_like(x) * s_noise
    sigma_hat = sig * (gamma+1)
    if gamma > 0: x = x + eps * (sigma_hat**2-sig**2)**0.5
    denoised = denoise(model, x, sig)
    d = (x-denoised)/sig
    dt = sig2-sigma_hat
    x_2 = x + d*dt
    if sig2==0: return x_2
    denoised_2 = denoise(model, x_2, sig2)
    d_2 = (x_2-denoised_2)/sig2
    d_prime = (d+d_2)/2
    return x + d_prime*dt


def linear_multistep_coeff(order, t, i, j):
    if order-1 > i: raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k: continue
            prod *= (tau-t[i-k]) / (t[i-j]-t[i-k])
        return prod
    return integrate.quad(fn, t[i], t[i+1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, sz = (2048,1,32,32), steps=100, order=4, sigma_max=80.):
    preds = []
    x = torch.randn(sz).cuda()*sigma_max
    sigs = sigmas_karras(steps, sigma_max=sigma_max)
    ds = []
    for i in range(len(sigs)-1):
        sig = sigs[i]
        denoised = denoise(model, x, sig)
        d = (x-denoised)/sig
        ds.append(d)
        if len(ds) > order: ds.pop(0)
        cur_order = min(i+1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigs, i, j) for j in range(cur_order)]
        x = x + sum(coeff*d for coeff, d in zip(coeffs, reversed(ds)))
        preds.append(x)
    return preds
"""


"""
def noisify(x0):
    device = x0.device
    sig = (torch.randn([len(x0)]) * 1.2 - 1.2).exp().to(x0).reshape(-1,1,1,1)
    noise = torch.randn_like(x0, device=device)
    c_skip,c_out,c_in = scalings(sig)
    noised_input = x0 + noise*sig
    target = (x0-c_skip*noised_input)/c_out
    return (noised_input*c_in,sig.squeeze()),target
"""


def scalings(sig: torch.Tensor, sigma_data: float):
    total_variance = sig ** 2 + sigma_data ** 2
    # c_skip, c_out, c_in
    return sigma_data ** 2 / total_variance, sig * sigma_data / total_variance.sqrt(), 1 / total_variance.sqrt()


def time_steps_lognorm_fn(number_of_samples: int, device='cpu', dtype=torch.float32):
    steps = (torch.randn([number_of_samples], dtype=dtype, device=device) * 1.2 - 1.2).exp()
    return steps


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def sigmas_karras(nb_steps=200, sigma_min=0.01, sigma_max=80.0, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, nb_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def denoise(model, x, sig, sigma_data, scalings_fn, **model_kwargs):
    sig = sig[None] #* torch.ones((len(x),1), device=x.device)
    c_skip, c_out, c_in = scalings_fn(sig, sigma_data)
    return model(x * c_in, sig, **model_kwargs) * c_out + x * c_skip


@torch.no_grad()
def sample_euler(model, x, sigs, i, sigma_data, scalings_fn, **model_kwargs):
    sig, sig2 = sigs[i], sigs[i+1]
    denoised = denoise(model, x, sig, sigma_data, scalings_fn=scalings_fn, **model_kwargs)
    return x + (x-denoised) / sig *(sig2 - sig)


def sample(model, x_T, sampler, sigmas, sigma_data, scalings_fn, **model_kwargs):
    x = x_T
    for i in range(len(sigmas) - 1):
        x = sampler(model, x, sigmas, i, sigma_data=sigma_data, scalings_fn=scalings_fn, **model_kwargs)
    return x


def loss_weighting_soft_min_snr(sigma, sigma_data):
    return (sigma * sigma_data) ** 2 / (sigma ** 2 + sigma_data ** 2) ** 2


def loss_weighting_snr(sigma, sigma_data):
    return sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)


def loss_weighting_karras(sigma, sigma_data):
    return torch.ones_like(sigma)


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = torch.rand(shape, device=device, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, device=device, dtype=dtype) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an uniform distribution."""
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


class KarrasDiffusion(Model):
    """
    Based on:
        https://github.com/fastai/course22p2/blob/master/nbs/26_diffusion_unet.ipynb
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            sigma_data: float,
            sigmas_train_sampling_fn: Callable[[int, int], int] = rand_log_normal,
            sample_fn = partial(sample, sampler=sample_euler),
            scalings_fn = scalings,
            sigmas_fn = sigmas_karras,
            loss_weighting = loss_weighting_soft_min_snr,
            loss_fn = loss_mse):
        """
        """
        super().__init__()
        self.model = model
        self.sigmas_train_sampling_fn = sigmas_train_sampling_fn
        self.sample_fn = sample_fn
        self.loss_fn = loss_fn
        self.scalings_fn = scalings_fn
        self.sigma_data = sigma_data
        self.sigmas_fn = sigmas_fn
        self.loss_weighting = loss_weighting


    def sample(self, batch_shape: Sequence[int], x_T: Optional[TorchTensorNX] = None, **model_extra_kwargs) -> TorchTensorNX:
        device = next(iter(self.model.parameters())).device

        if x_T is None:
            x_T = torch.randn(batch_shape, device=device)
        else:
            x_T = x_T.to(device)

        sigmas = self.sigmas_fn().to(device)
        sigma_max = sigmas[0]
        x_T = x_T * sigma_max
        return self.sample_fn(model=self.model, x_T=x_T, sigmas=sigmas, sigma_data=self.sigma_data, scalings_fn=self.scalings_fn, **model_extra_kwargs)


    def _loss(self, output, target, sigmas):
        loss_per_sample = self.loss_fn(output, target).flatten(1).mean(1)
        weighted_loss_per_sample = loss_per_sample * self.loss_weighting(sigmas, self.sigma_data)
        return weighted_loss_per_sample


    def training_step(self, x0: TorchTensorNX, **model_extra_kwargs) -> Dict[str, TorchTensorNX]:
        device = x0.device
        sigmas = self.sigmas_train_sampling_fn(len(x0), device=device)
        assert sigmas.shape == (len(x0),)
        noise = torch.randn_like(x0, device=device)

        c_skip, c_out, c_in = self.scalings_fn(sigmas, self.sigma_data)
        c_in = expand_dim_like(c_in, x0)
        c_skip = expand_dim_like(c_skip, x0)
        c_out = expand_dim_like(c_out, x0)

        noised_input = x0 + noise * expand_dim_like(sigmas, x0)
        target = (x0 - c_skip * noised_input) / c_out
        
        model_output = self.model(noised_input * c_in, sigmas, **model_extra_kwargs)
        loss = self._loss(model_output, target, sigmas)
        return {'loss': loss, 'timesteps': sigmas} 
