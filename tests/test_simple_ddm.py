import torch

from flextrain.diffusion.discrete_ddpm_simple import SimpleGaussianDiffusion

if __name__ == '__main__':
    model = torch.nn.Linear(1, 1)
    ddpm = SimpleGaussianDiffusion(model)

    x0 = torch.zeros(10, 1, 64, 64)

    t = ddpm.sample_timesteps(x0.shape[0])
    x_t = ddpm.forward_diffusion(x0, t)

    print('DONE')
