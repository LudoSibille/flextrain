from flextrain.diffusion.discrete_beta_schedulers import (
    get_ᾱ_from_β,
    linear,
    linear_zero_snr,
)


def test_linear_zero_snr():
    """
    Test properties of zero SNR noise schedulers
    """

    # Signal-to-noise ratio (SNR) can be calculated as
    # SNR(t) = ᾱt / (1 - ᾱt)
    # ensure ᾱt = 0 for t = T
    for nb_steps in [100, 1000, 10000]:
        betas = linear_zero_snr(steps=nb_steps, β_end=0.02)
        assert betas.shape == (nb_steps,)
        ᾱ = get_ᾱ_from_β(betas)
        ᾱ_sqrt = ᾱ.sqrt()
        assert abs(ᾱ_sqrt[-1]) < 1e-6


def test_scheduler_linear_epoch_invariance():
    """
    scheduler must be invariant to the number of epochs (i.e., the noising process
    should be not be affected by different steps used)
    """
    betas_1000 = linear(steps=1000)
    betas_10000 = linear(steps=10000)
    betas_10000_10 = betas_10000[::10]
    assert (betas_1000 - betas_10000_10).abs().max() < 2e-2
