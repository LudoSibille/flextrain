import os
from .git import get_git_revision, run_command
import json
import logging
import sys
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from .callback import Callback
from mdutils import MdUtils


logger = logging.getLogger(__name__)


class CallbackExperimentTracing(Callback):
    """
    Record useful information to track the experiments (e.g., git hash, git status, python environment)
    so that we can easily reproduce the results
    """
    def __init__(self, output_dir_name='tracing'):
        super().__init__()
        self.output_dir_name = output_dir_name
        self.output_path = None

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        logger.info('CallbackExperimentTracing')
        if self.output_path is None:
            self.output_path = os.path.join(trainer.options.data.root_current_experiment, self.output_dir_name)
            os.makedirs(self.output_path, exist_ok=True)

        git_revision = get_git_revision()
        with open(os.path.join(self.output_path, 'git_revision.json'), 'w') as f:
            json.dump({'git_revision': git_revision}, f, indent=3)

        here = os.path.abspath(os.path.dirname(__file__))
        git_status = run_command(['git', 'status'], cwd=here)
        with open(os.path.join(self.output_path, 'git_status.txt'), 'w') as f:
            f.write(git_status)

        requirements = run_command(['pip', 'list', '--format', 'freeze'])
        with open(os.path.join(self.output_path, 'requirements.txt'), 'w') as f:
            f.write(requirements)

        with open(os.path.join(self.output_path, 'runtime_config.txt'), 'w') as f:
            f.write(str(sys.argv))

        logger.info('CallbackExperimentTracing done!')


    def make_markdown_report(self, md: MdUtils, base_level: int = 1):
        md.new_header(level=base_level, title=f'Tracing')

        path = os.path.join(self.output_path, 'git_revision.json')
        if os.path.exists(path):
            with open(path) as f:
                lines = f.read()
            md.new_header(level=base_level + 1, title=f'Git Revisions')
            md.new_line(lines)    
        
        files = ('git_revision.json', 'git_status.json', 'runtime_config.json', 'requirements.txt')
        headers = ('Git Revisions', 'Git status', 'Runtime config', 'Requirements')

        for header, file in zip(headers, files):
            path = os.path.join(self.output_path, file)
            if os.path.exists(path):
                with open(path) as f:
                    lines = f.read()
                md.new_header(level=base_level + 1, title=header)
                md.new_line(lines)    
