import os
import logging
import subprocess


logger = logging.getLogger(__name__)


def run_command(args, cwd=None) -> str:
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=cwd
        )

        stdout, stderr = process.communicate()
        stdout = stdout.decode('utf-8').strip()
        stderr = stderr.decode('utf-8').strip()
    except Exception as e:
        return f'Command failed={args}, Exception={e}, cwd={cwd}'

    if len(stderr) == 0:
        return stdout
    else:
        return f'{stdout}\n<STDERR>\n{stderr}'


def get_git_revision():
    """
    Return the current git revision
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return run_command(['git', 'rev-parse', 'HEAD'], cwd=here)
