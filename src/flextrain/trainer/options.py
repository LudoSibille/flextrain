from types import SimpleNamespace
from typing import Literal, Optional, Union


class SimpleRepr(object):
    """
    A mixin implementing a simple __repr__.
    """

    def __repr__(self) -> str:
        return "<{klass} {attrs}>".format(
            klass=self.__class__.__name__,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )


class Training(SimpleRepr):
    def __init__(
        self,
        nb_epochs: int = 100,
        precision: Union[int, Literal['16', '32', '16-mixed']] = 32,
        pretraining: str = '',
        accumulate_grad_batches: int = 1,
        accelerator: str = 'gpu',
        checkpoint: Optional[str] = None,
        check_val_every_n_epoch: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[Literal['value', 'norm']] = None,
        enable_checkpointing: Optional[bool] = None,
        detect_anomaly: bool = False,
    ) -> None:
        self.devices = '0'
        self.nb_epochs = nb_epochs
        self.precision = precision
        self.pretraining = pretraining
        self.accumulate_grad_batches = accumulate_grad_batches
        self.accelerator = accelerator
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.checkpoint = checkpoint
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.enable_checkpointing = enable_checkpointing
        self.detect_anomaly = detect_anomaly


class Data(SimpleRepr):
    def __init__(self, root_current_experiment: Optional[str] = None, with_script_directory_prefix: bool = True) -> None:
        self.root_current_experiment = root_current_experiment
        self.with_script_directory_prefix = with_script_directory_prefix


class Workflow(SimpleRepr):
    def __init__(
        self,
        enable_progress_bar: bool = True,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
    ) -> None:
        self.enable_progress_bar = enable_progress_bar
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches


class Options(SimpleRepr):
    def __init__(
        self,
        model: SimpleNamespace = SimpleNamespace(),
        training: Training = Training(),
        workflow: Workflow = Workflow(),
        data: Data = Data(),
    ) -> None:

        self.model = model
        self.training = training
        self.workflow = workflow
        self.data = data
