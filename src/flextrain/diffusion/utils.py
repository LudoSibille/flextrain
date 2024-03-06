import torch
import functools
import logging


def expand_dim_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Expand the dimension of a tensor to a target dimension. The missing dimensions are added as size `1`
    """
    source_dim = len(source.shape)
    target_dim = len(target.shape)
    if target_dim > source_dim:
        source_shape_target = list(source.shape) + [1] * (target_dim - source_dim)
        return source.view(source_shape_target)
    
    if target_dim == source_dim:
        return source
    
    raise ValueError(f'source dimension should be higher than target dimension! source_dim={source_dim}, target_dim={target_dim}')


def catch_all_and_log(f):
    """
    Decorator to catch all exception and log the error
    """
    @functools.wraps(f)
    def inner(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logging.exception('catch_all_and_log: caught exception!')
            return None
    
    return inner