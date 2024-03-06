from typing import Sequence
from torch import nn
from functools import partial


def make_encoder(
        input_size: int, 
        layer_sizes: Sequence[int], 
        norm: nn.Module = nn.InstanceNorm2d, 
        act: nn.Module = nn.LeakyReLU, 
        norm_last: bool = True, 
        kernel_size: int = 3, 
        pool: Sequence[int] = (),
        pool_fn: nn.Module = partial(nn.MaxPool2d, kernel_size=2)):
    layers = [nn.Conv2d(input_size, layer_sizes[0], kernel_size=kernel_size, padding=kernel_size // 2)]

    last = layer_sizes[0]
    for size_n, size in enumerate(layer_sizes):
        layers += [
            nn.Conv2d(last, size, kernel_size=kernel_size, padding=kernel_size // 2),
            act(),
            norm(size),
        ]

        if size_n in pool:
            layers.append(pool_fn())

        last = size
    
    if not norm_last:
        layers.pop()

    return nn.Sequential(*layers)


def make_decoder(
        output_size: int, 
        layer_sizes: Sequence[int], 
        norm: nn.Module = nn.InstanceNorm2d, 
        act: nn.Module = nn.LeakyReLU, 
        kernel_size: int = 3, 
        unpool: Sequence[int] = (),
        unpool_fn=partial(nn.UpsamplingNearest2d, scale_factor=2)):
    
    layers = []
    last = layer_sizes[0]
    for size_n, size in enumerate(layer_sizes[1:]):
        layers += [
            nn.Conv2d(last, size, kernel_size=kernel_size, padding=kernel_size // 2),
            act(),
            norm(size),
        ]

        if size_n in unpool:
            layers.append(unpool_fn())
        last = size

    return nn.Sequential(*layers, nn.Conv2d(last, output_size, kernel_size=kernel_size, padding=kernel_size // 2))