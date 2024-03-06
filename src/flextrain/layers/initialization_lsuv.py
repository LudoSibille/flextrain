from typing import Any, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..trainer.utils import transfer_batch_to_device
from .activations import GeneralReLU
from ..types import Batch
import logging
logger = logging.getLogger(__name__)


class HookMeanStd:
    def __init__(self, m: nn.Module) -> None:
        self.means = []
        self.stds = []
        self.hook = m.register_forward_hook(self.hook_fn)

    @torch.no_grad()
    def hook_fn(self, module: nn.Module, inputs: Any, output: torch.Tensor):
        std, mean = torch.std_mean(output)
        self.means.append(mean)
        self.stds.append(std)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def clear(self):
        self.means = []
        self.stds = []

    def __del__(self):
        self.remove()


@torch.no_grad()
def lsuv_init(model, m_act: nn.Module, m_in: nn.Module, batches: Sequence[Batch], nb_tries: int = 20, tol_mean=1e-2, tol_std=1e-2):
    hook = HookMeanStd(m_act)
    for _ in range(nb_tries):
        for batch in batches:
            model(batch)

        if len(hook.means) == 0:
            # No hook found!
            logger.error(f'no activation recorded for this layer: {m_act}, {m_in}')
            return
        mean = torch.mean(torch.tensor(hook.means))
        std = torch.mean(torch.tensor(hook.stds))
        if abs(mean) < tol_mean and abs(1 - std) < tol_std:
            # layer is initialized
            hook.remove()
            return
        
        m_in.bias.data -= mean
        m_in.weight.data /= std
        hook.means.clear()
        hook.stds.clear()
    
    print(f'LSUV layer={type(m_act).__name__}, Mean={mean}, std={std}')
    logger.info(f'LSUV layer={type(m_act).__name__}, Mean={mean}, std={std}')

    # exhausted the number of attempts
    hook.remove()


def lsuv_(
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device, 
        nb_batches: int = 1, 
        norm_fn: Optional[nn.Module] = lambda m: nn.init.orthogonal_(m.weight.data) if hasattr(m, 'weight') and m.weight.ndimension() >= 2 else None,
        activations_target: Tuple[nn.Module] = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.CELU, GeneralReLU, nn.Softplus, nn.SELU),
        init_layers_target: Tuple[nn.Module] = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear),
        last_layer_no_activation: bool = True):
    """
    Layer-sequential unit-variance (LSUV) initialization

    Initialize model such that each layer has a (0-mean, 1-std) initialization

    https://arxiv.org/pdf/1511.06422.pdf

    """
    if norm_fn:
        model.apply(norm_fn)

    model = model.to(device)
    activations = [o for o in model.modules() if isinstance(o, activations_target)]
    layers = [o for o in model.modules() if isinstance(o, init_layers_target)]

    if last_layer_no_activation:
        assert len(activations) + 1 == len(layers), 'is the last layer with activation? If true, set last_layer_no_activation=False'
        # the output should be the layer, there is no activation function!
        activations.append(layers[-1])

    batches = []
    for batch in dataloader:
        batches.append(batch)
        if len(batches) >= nb_batches:
            break

    dataloader = [transfer_batch_to_device(b, device) for b in dataloader]
    for act, layer in zip(activations, layers):
        lsuv_init(model, act, layer, dataloader)