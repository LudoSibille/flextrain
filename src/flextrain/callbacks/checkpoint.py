import logging
from typing import Callable, Dict, Optional
import lightning as L
import os
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from ..trainer.utils import to_value

logger = logging.getLogger(__name__)


def save_model_no_NaN(callback, trainer: L.Trainer, pl_module: L.LightningModule, train_only: bool = True) -> bool:
    """
    Save model ONLY if there are not NaNs in metrics (optionally, only in training)
    """

    # check the model has any parameter of buffer with NaN
    nan_buffers = [torch.isnan(p).any() for p in pl_module.buffers()]
    if len(nan_buffers) > 0:
        model_nan_buffers = torch.stack(nan_buffers).sum()
        if model_nan_buffers:
            logger.error(f'model has NaN in buffers. N={model_nan_buffers}')
            return False
        
    nan_parameters = [torch.isnan(p).any() for p in pl_module.parameters()]
    if len(nan_parameters) > 0:
        model_nan_parameters = torch.stack(nan_parameters).sum()
        if model_nan_parameters:
            logger.error(f'model has NaN in parameters. N={model_nan_parameters}')
            return False

    for name, value in trainer.callback_metrics.items():
        if train_only and 'train' not in name:
            continue

        if np.isnan(to_value(value)):
            return False
    
    return True


class CallbackCheckpoint(L.Callback):
    def __init__(
            self, 
            save_top_k: int = 3,
            output_folder: str = 'checkpoints',
            metric_name: Optional[str] = None,
            metric_to_maximize: Optional[bool] = None,
            save_last: bool = True,
            save_model_check_fn: Callable[["CallbackCheckpoint", L.Trainer, L.LightningModule], bool] = save_model_no_NaN,
            restore_last_valid_checkpoint_on_NaN: bool = True
            ) -> None:
        
        super().__init__()
        self.save_top_k = save_top_k
        self.output_folder = output_folder
        self.metric_name = metric_name
        self.metric_to_maximize = metric_to_maximize
        if self.metric_name is not None:
            assert metric_to_maximize is not None, f'metric={metric_name} is defined. Is this to be maximized or minimized? This must be specified!'
        self.save_last = save_last
        self.last_checkpoints = []
        self.last_checkpoints_metric = []
        self.save_model_check_fn = save_model_check_fn
        self.restore_last_valid_checkpoint_on_NaN = restore_last_valid_checkpoint_on_NaN

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.info('Checkpoint check')

        path_root = os.path.join(trainer.options.data.root_current_experiment, self.output_folder)
        os.makedirs(path_root, exist_ok=True)

        # check if model satisfy conditions
        save_model = self.save_model_check_fn(self, trainer, pl_module)
        if not save_model:
            logger.info('Check `self.save_model_check_fn` returned False. Model not saved!')
            if self.restore_last_valid_checkpoint_on_NaN and len(self.last_checkpoints) > 0:
                checkpoint_path = self.last_checkpoints[-1]
                logger.info(f'model restored from={checkpoint_path}')
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=pl_module.device)
                pl_module.load_state_dict(checkpoint)
            return
        
        # export the last model
        path_last = os.path.join(path_root, 'last.pth')
        with open(path_last, 'wb') as f:
            torch.save(pl_module.state_dict(), f)

        metric_value = None
        if self.metric_name is not None:
            metric_value = to_value(trainer.callback_metrics.get(self.metric_name))

        if len(self.last_checkpoints) >= self.save_top_k:
            # remove one check point
            if self.metric_name is not None:
                if metric_value is None:
                    logger.info(f'metric={self.metric_name} not found in the metrics. Available={list(trainer.callback_metrics.keys())}. Top-k not recorded!')
                    return

                # find the worst metric
                if self.metric_to_maximize:
                    index = np.argmin(self.last_checkpoints_metric)
                    if self.last_checkpoints_metric[index] > metric_value:
                        # metric is worst
                        return
                else:
                    index = np.argmax(self.last_checkpoints_metric)
                    if self.last_checkpoints_metric[index] < metric_value:
                        # metric is worst
                        return
            else:
                index = 0
            
            # remove the worst checkpoint
            try:
                os.remove(self.last_checkpoints[index])
            except:
                pass
            del self.last_checkpoints[index]
            del self.last_checkpoints_metric[index]

        if metric_value is not None:
            path_checkpoint = os.path.join(path_root, f'checkpoint_{trainer.current_epoch}_{self.metric_name}_{metric_value}.pth')
        else:
            path_checkpoint = os.path.join(path_root, f'checkpoint_{trainer.current_epoch}.pth')

        with open(path_checkpoint, 'wb') as f:
            torch.save(pl_module.state_dict(), f)

        self.last_checkpoints.append(path_checkpoint)
        self.last_checkpoints_metric.append(metric_value)
        logger.info('Checkpoint done!')
