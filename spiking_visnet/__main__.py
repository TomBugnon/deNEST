#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __main__.py

"""
Spiking VisNet
~~~~~~~~~~~~~~

Usage:
    python -m spiking_visnet <param_file.yml> [options]
    python -m spiking_visnet -h | --help
    python -m spiking_visnet -v | --version

Arguments:
    <param_file.yml>  YAML file containing simulation parameters.

Options:
    -o --output=PATH  Directory in which simulation results will be saved.
                      Overrides <param_file.yml>.
    -i --input=PATH   Path to a stimuli array to present to the network during
                      each session. Overrides <param_file.yml>.
    -h --help         Show this.
    -v --version      Show version.
"""

import random
import sys

from docopt import docopt

from config import PYTHON_SEED

from . import run
from .__about__ import __version__
from .parameters import Params

random.seed(PYTHON_SEED)

# Maps CLI options to their corresponding path in the parameter tree.
_CLI_ARG_MAP = {
    '<param_file.yml>': ('children', 'simulation', 'param_file_path'),
    '--input': ('children', 'sessions', 'params', 'user_input'),
    '--output': ('children', 'simulation', 'user_savedir'),
}


def main():
    """Run Spiking VisNet from the command line."""
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m spiking_visnet` calling pattern.
    argv = ['-m', 'spiking_visnet'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv, version=__version__)
    # Get parameter overrides from the CLI options.
    overrides = Params({_CLI_ARG_MAP[key]: value
                        for key, value in arguments.items()
                        if key in _CLI_ARG_MAP})
    # Run it!
    run(arguments['<param_file.yml>'], overrides=overrides)


if __name__ == '__main__':
    main()
