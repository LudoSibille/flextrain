import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
import time
from ..trainer.utils import to_value
import logging


logger = logging.getLogger(__name__)


def format_float(value):
    return f'{to_value(value):.5f}'


class CallbackLogMetrics(L.Callback):
    """
    Log important metrics for each epoch
    """
    def __init__(self) -> None:
        super().__init__()
        self.lowest_metrics = {}
        self.nb_train_batches = 0

        self.total_time_train_batch = 0
        self.time_train_batch_start = 0

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        rank_zero_info(f'started epoch={trainer.current_epoch}')
        self.epoch_start_time = time.perf_counter()
        self.total_time_train_batch = 0

    @rank_zero_only
    def on_train_batch_start(self, trainer: L.Trainer, *args, **kwargs) -> None:
        self.nb_train_batches += 1
        self.time_train_batch_start = time.perf_counter()

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs) -> None:
        self.total_time_train_batch += time.perf_counter() - self.time_train_batch_start

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_end_time = time.perf_counter()

        lr = pl_module.optimizers().param_groups[0]['lr']

        log_str = f'end epoch={trainer.current_epoch}, time={self.epoch_end_time - self.epoch_start_time:.2f}, nb_train_batches={self.nb_train_batches}, total_train_batch_time={self.total_time_train_batch:.2f}, LR={lr}'
        rank_zero_info(log_str)
        logger.info(log_str)
        self.nb_train_batches = 0
        for key, value in trainer.callback_metrics.items():
            if '_step' == key[-5:]:
                # not interested in step metric
                continue
            if key + '_epoch' in trainer.callback_metrics:
                # duplicate
                continue

            if key not in self.lowest_metrics:
                self.lowest_metrics[key] = (value, trainer.current_epoch)

            lowest_value, lowest_epoch = self.lowest_metrics.get(key)
            if lowest_value > value:
                self.lowest_metrics[key] = (value, trainer.current_epoch)
                # update for the current epoch
                lowest_value, lowest_epoch = self.lowest_metrics.get(key)

            if key not in ["log", "progress_bar"]:
                log_str = f'   {key}={format_float(value)} [best={format_float(lowest_value)}, epoch={lowest_epoch}]'
                rank_zero_info(log_str)
                logger.info(log_str)