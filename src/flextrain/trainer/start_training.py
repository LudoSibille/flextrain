import logging
import os
import sys
from pprint import pformat
from typing import Dict, List, Sequence, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from mdutils.mdutils import MdUtils

from ..callbacks.callback import Callback
from .options import Options
from .utils import create_or_recreate_folder, default, is_debug_run

logger = logging.getLogger(__name__)


def get_experiment_root(options: Options, with_script_directory_prefix: bool = True) -> str:
    prefix = ''
    if with_script_directory_prefix:
        prefix = os.path.basename(os.path.dirname(os.path.abspath(sys.argv[0])))

    if options.data.root_current_experiment is None:
        root = default('EXPERIMENT_ROOT', default_value=None)
        if root is None:
            if os.name == 'nt':
                return f'c:/tmp/experiments/{prefix}'
            return f'/tmp/experiments/{prefix}'
        else:
            return os.path.join(root, prefix)

    return options.data.root_current_experiment


@rank_zero_only
def setup_folders(experiment_folder: str, options: Options) -> str:
    # only the global rank 0 should have a logger setup!
    is_debug = is_debug_run()

    if not is_debug:
        # protect against unfortunate experiment re-run and previous results destruction
        assert not os.path.exists(experiment_folder), (
            f'experiment_folder={experiment_folder} ' 'already exists! It must be manually removed!'
        )

    # only rank 0 is responsible for logging in DDP
    create_or_recreate_folder(experiment_folder)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(experiment_folder, 'trainer_logging_rank.log'),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        level=logging.INFO,
        filemode='w',
    )
    logger.info('Training started!')

    # redirect the lightning logs to a file
    logger_lightning = logging.getLogger('lightning.pytorch')
    logging.getLogger('lightning.pytorch').setLevel(logging.INFO)
    logger_lightning.addHandler(logging.FileHandler(os.path.join(experiment_folder, 'pytorch_lightning_rank.log')))
    print('logging initialized!')

    options_str = pformat(options, indent=3)
    logger_lightning.info(f'options=\n{options_str}')
    logger.info(f'options=\n{options_str}')
    return experiment_folder


def load_model(
    pretraining: str,
    model_pl: torch.nn.Module,
    strict: bool = True,
    discard_keys: Sequence[str] = ('.dice.'),
) -> Tuple[List[str], List[str]]:
    """
    Load model from a file.

    Parameters:
        discard_keys: if `strict`, relax this constraint by allowing some known keys not to be loaded.
            (e.g., `monai.loss.dice.Dice` class weight)
    """
    model_state = torch.load(pretraining, map_location=torch.device('cpu'))
    if 'state_dict' in model_state:
        # Lightning callback export
        state_dict = model_state['state_dict']
    else:
        # typical manual export
        state_dict = model_state
    missing_keys, additional_keys = model_pl.load_state_dict(state_dict, strict=False)
    if strict:
        assert len(additional_keys) == 0, f'additional keys={additional_keys}'
        for k in missing_keys:
            to_discard = False
            for d in discard_keys:
                if d in k:
                    to_discard = True
                    break
            assert to_discard, f'key not found={k}, all={missing_keys}'

    return missing_keys, additional_keys


def start_training(
    options: Options,
    datasets_loaders: Dict,
    callbacks: Sequence[L.Callback],
    model_pl: L.LightningModule,
) -> None:
    logger.info('Training Starting...')
    experiment_root = get_experiment_root(options, with_script_directory_prefix=options.data.with_script_directory_prefix)
    root_current_experiment = os.path.join(experiment_root, os.path.basename(sys.argv[0]).replace('.py', ''))
    if is_debug_run():
        # treat debug run differently: typically, very short
        # run to excercise the algorithm but not to perform the
        # full experiment
        # DO NOT do this in `setup_folders` (multi-processes issues!)
        root_current_experiment = root_current_experiment + '_vs'
    options.data.root_current_experiment = root_current_experiment
    setup_folders(root_current_experiment, options)

    logging.info(f'Experiment root={root_current_experiment}')
    print(f'-------- Experiment root={root_current_experiment} --------')
    logger_tb = L.pytorch.loggers.TensorBoardLogger(save_dir=options.data.root_current_experiment)
    logger_csv = L.pytorch.loggers.CSVLogger(save_dir=options.data.root_current_experiment)

    # load only the state of the model, NOT the state of the training!
    if (
        hasattr(options.training, 'pretraining')
        and options.training.pretraining is not None
        and len(options.training.pretraining) > 0
    ):
        load_model(options.training.pretraining, model_pl)

    if hasattr(options.workflow, 'limit_train_batches'):
        limit_train_batches = options.workflow.limit_train_batches
    else:
        limit_train_batches = None

    if hasattr(options.workflow, 'limit_val_batches'):
        limit_val_batches = options.workflow.limit_val_batches
    else:
        limit_val_batches = None

    L.pytorch.seed_everything(workers=True)
    trainer = L.Trainer(
        accumulate_grad_batches=int(options.training.accumulate_grad_batches),
        accelerator=options.training.accelerator,
        gradient_clip_val=options.training.gradient_clip_val,
        # force using a specific device
        devices=[int(d) for d in options.training.devices.split(',')],
        logger=[logger_csv, logger_tb],
        max_epochs=int(options.training.nb_epochs),
        default_root_dir=options.data.root_current_experiment,
        precision=options.training.precision,  # type: ignore
        callbacks=callbacks,  # type: ignore
        num_sanity_val_steps=0,
        # amp_backend='native',
        enable_progress_bar=options.workflow.enable_progress_bar,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False),
        strategy=L.pytorch.strategies.ddp.DDPStrategy(find_unused_parameters=True),
        enable_model_summary=False,  # we will have one configured in the callbacks,
        check_val_every_n_epoch=options.training.check_val_every_n_epoch,
        enable_checkpointing=options.training.enable_checkpointing,
        detect_anomaly=options.training.detect_anomaly,
        gradient_clip_algorithm=options.training.gradient_clip_algorithm,
    )

    trainer.options = options  # type: ignore

    # possibly reload ALL the states (model & training)
    checkpoint = options.training.checkpoint if hasattr(options.training, 'checkpoint') else None

    # TODO handle multi datasets?
    assert len(datasets_loaders) == 1, f'TODO multi-datasets not handled yet. Datasets={datasets_loaders.keys()}'
    dataset_name = next(iter(datasets_loaders.keys()))

    trainer.fit(
        model=model_pl,
        train_dataloaders=datasets_loaders[dataset_name].get('train'),
        val_dataloaders=datasets_loaders[dataset_name].get('valid'),
        ckpt_path=checkpoint,
    )

    logger.info('Training DONE!')

    md_report_path = os.path.join(root_current_experiment, 'experiment_report.md')
    md = MdUtils(file_name=md_report_path, title='Report')
    for callback in callbacks:
        if isinstance(callback, Callback):
            try:
                callback.make_markdown_report(md, base_level=1)
            except Exception as e:
                logger.exception(f'Callback report failed={callback}', e)
    md.create_md_file()
    logger.info(f'Report exported={md_report_path}!')
