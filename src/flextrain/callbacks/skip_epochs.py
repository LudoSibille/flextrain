import lightning as L
from typing import Optional, Sequence
from .callback import Callback
import logging
from mdutils import MdUtils


logger = logging.getLogger(__name__)


class CallbackSkipEpoch(Callback):
    """
    Call the callbacks every few epochs and remap the events
    """
    def __init__(
            self, 
            callbacks: Sequence[L.Callback],
            nb_epochs: int, 
            include_epoch_zero: bool = False, 
            method_name: str = 'on_train_epoch_end', 
            mapping_output: Optional[str] = None, 
            on_global_zero_only=True) -> None:
        
        super().__init__()
        assert nb_epochs >= 1
        self.nb_epochs = nb_epochs
        m = getattr(self, method_name)
        assert m is not None, f'unhandled method={method_name}!'
        setattr(self, method_name, self._run)
        self.callbacks = callbacks
        self.method_name = method_name
        self.include_epoch_zero = include_epoch_zero
        self.on_global_zero_only = on_global_zero_only
        if mapping_output is None:
            self.mapping_output = method_name
        else:
            self.mapping_output = mapping_output


    def _run(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        last_epoch = trainer.current_epoch == trainer.options.training.nb_epochs - 1
        if not last_epoch:
            # always run the callbacks on the last epoch
            if self.on_global_zero_only and not trainer.is_global_zero:
                return

            if not self.include_epoch_zero and trainer.current_epoch == 0:
                return 
        
        if (trainer.current_epoch % self.nb_epochs == 0) or last_epoch:
            for c in self.callbacks:
                m = getattr(c, self.mapping_output)
                m(trainer, pl_module)


    def make_markdown_report(self, md: MdUtils, base_level: int = 1):
        for callback in self.callbacks:
            if isinstance(callback, Callback):
                try:
                    callback.make_markdown_report(md, base_level=base_level)
                except Exception as e:
                    logger.exception(f'Callback report failed={callback}', e)
