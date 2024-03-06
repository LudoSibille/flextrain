import time

import torch
from scipy.linalg import sqrtm

from flextrain.metrics.sqrtm import sqrtm_newton_schulz


def test_sqrtm_vs_scipy():
    torch.random.manual_seed(0)
    device = torch.device('cpu')
    for i in range(10):
        samples = torch.randn([10000, 1000 + i], dtype=torch.float64)

        # calculate the correlation matrix: ensures this is positive semi definite matrix
        correlation = samples.T @ samples
        correlation = correlation.to(device)
        time_start = time.perf_counter()
        correlation_sqrt = sqrtm_newton_schulz(correlation, tol=1e-9)
        time_end = time.perf_counter()
        correlation = correlation.to(torch.device('cpu'))
        correlation_sqrt = correlation_sqrt.to(torch.device('cpu'))

        max_error = (correlation - correlation_sqrt @ correlation_sqrt).abs().max()
        assert max_error < 1e-5, f'max_error={max_error}'

        time_scipy_start = time.perf_counter()
        correlation_sqrt_scipy = sqrtm(correlation)
        time_scipy_end = time.perf_counter()
        max_error = (correlation_sqrt - correlation_sqrt_scipy).abs().max()
        assert max_error < 5e-5, f'max_error={max_error}'

        print(correlation.shape, max_error)
        print('Time=', time_end - time_start, 'Time SCIPY=', time_scipy_end - time_scipy_start)
