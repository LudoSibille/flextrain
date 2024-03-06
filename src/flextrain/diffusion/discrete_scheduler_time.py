import torch
import math


def rand_log_uniform(number_of_samples, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(number_of_samples, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_log_logistic(number_of_samples, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(number_of_samples, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_normal(number_of_samples, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(number_of_samples, device=device, dtype=dtype) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def rand_uniform_steps(number_of_samples, max_value, device='cpu', dtype=torch.int32):
    return torch.randint(low=1, high=max_value, size=[number_of_samples], device=device, dtype=dtype)


def rand_log_uniform_steps(number_of_samples, max_value, device='cpu', dtype=torch.int32):
    return rand_log_uniform(number_of_samples, min_value=1, max_value=max_value, device=device).type(dtype)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=True)

    nb_samples = 100000
    #axs[0].hist(rand_log_uniform(nb_samples, 1e-7, 1.0))
    #axs[1].hist(rand_log_logistic(nb_samples, 1e-7, 1.0))
    #axs[2].hist(rand_log_normal(nb_samples, 1e-7, 1.0))
    #samples = 1.0 - torch.cos(torch.rand(nb_samples) * math.pi / 2.0)
    samples = rand_denoising_steps(nb_samples, 1000)
    axs.hist(samples)
    plt.savefig('/tmp/experiments/timesteps.png')
    pass