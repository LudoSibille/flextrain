import logging
from dataclasses import dataclass
from typing import Any, Sequence

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from torch import nn

from ..trainer.utils import number2human, transfer_batch_to_device
from .autoencoder import get_batch_iter

logger = logging.getLogger(__name__)


@dataclass
class Summary:
    total_parameters: int = 0
    total_flops = 0
    string: str = f'{"Name":20}|{"Input shape":40}|{"Output shape":25}|{"Params":10}|{"FLOPS":10}\n{"-" * (20+40+25+10+5)}\n'


def _flops(module: nn.Module, output):
    # very rough approximation...
    if isinstance(output, torch.Tensor):
        if output.dim() < 3:
            return output.numel()
        if output.dim() == 4:
            *_, h, w = output.shape
            return output.numel() * h * w
        if output.dim() == 5:
            *_, d, h, w = output.shape
            return output.numel() * h * w * d
    return None


def get_shape(output):
    if isinstance(output, list):
        os = [get_shape(o) for o in output]
        return os
    return tuple(output.shape)


class HookRecordModuleInfo:
    def __init__(self, m: nn.Module, output_summary: Summary) -> None:
        self.hook = m.register_forward_hook(self.hook_fn)
        self.output_summary = output_summary

    @torch.no_grad()
    def hook_fn(self, module: nn.Module, inputs: Any, output: torch.Tensor):
        input_shape = []
        for i in inputs:
            if isinstance(i, torch.Tensor):
                input_shape.append(str(tuple(i.shape)))
            else:
                input_shape.append(type(i).__name__)
        input_shape_str = ','.join(input_shape)
        nparams = sum(o.numel() for o in module.parameters())

        flops = _flops(module, output)
        if flops is None:
            flops = '???'
        else:
            self.output_summary.total_flops += flops
            flops = number2human(flops)

        layer_str = f'{type(module).__name__:20}|{input_shape_str:40}|{str(get_shape(output)):25}|{number2human(nparams):10}|{flops:10}\n'
        self.output_summary.string += layer_str
        self.output_summary.total_parameters += nparams

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __del__(self):
        self.remove()


class CallbackModelSummary(L.Callback):
    """
    Log important metrics for each epoch
    """

    def __init__(self, layers_to_discard=(nn.Sequential,)) -> None:
        super().__init__()
        self.layers_to_discard = layers_to_discard
        self.recorded = False

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.recorded:
            return

        self.recorded = True
        device = pl_module.device
        batch = next(get_batch_iter(trainer))
        batch = transfer_batch_to_device(batch, device)

        hooks = []
        output_summary = Summary()
        for m in pl_module.modules():
            if isinstance(m, self.layers_to_discard):
                # discard this layer
                continue

            hook = HookRecordModuleInfo(m, output_summary=output_summary)
            hooks.append(hook)

        try:
            pl_module.forward(batch)
        except Exception as e:
            logger.error('forward failed!')
            logger.exception(e)

        for hook in hooks:
            hook.remove()

        print(f'Model summary:\n{output_summary.string}')
        print(f'total trainable parameters={number2human(output_summary.total_parameters)}')
        print(f'total FLOPS={number2human(output_summary.total_flops)}')
        logger.info(f'Model summary:\n{output_summary.string}')
        logger.info(f'total trainable parameters={number2human(output_summary.total_parameters)}')
        logger.info(f'total FLOPS={number2human(output_summary.total_flops)}')
