#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/__main__.py
"""
preprocessing.

~~~~~~~~~~~~~~

Usage:
    python -m nets.preprocess <preprocessing_params> <sim_params> [-i <input_path>]
    python -m nets.preprocess -h | --help

Arguments:
    <preprocessing_params>  Relative path to preprocessing parameter yaml file
    <sim_params>            Relative path to full simulation parameter file

Options:
    -i --input=<input_path>  Input directory. If not specified, uses INPUT_PATH from config.
    -h --help               Show this
"""

import random
import sys

from config import PYTHON_SEED
from docopt import docopt
from user_config import INPUT_PATH

from . import run
from ..utils.autodict import AutoDict
from ..utils.structures import dictify

random.seed(PYTHON_SEED)

# Maps CLI options to their corresponding path in the parameter tree.
_SIM_CLI_ARG_MAP = {
    '<sim_params>': ('children', 'simulation', 'param_file_path')
}
_PREPRO_CLI_ARG_MAP = {}


def main():
    """Preprocess inputs from the command line."""
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m nets` calling pattern.
    argv = ['-m', 'nets.preprocess'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv)
    if not arguments['--input']:
        arguments['--input'] = INPUT_PATH
    sim_overrides = dictify(
        AutoDict({
            _SIM_CLI_ARG_MAP[key]: value
            for key, value in arguments.items()
            if key in _SIM_CLI_ARG_MAP
        }))
    prepro_overrides = dictify(
        AutoDict({
            _PREPRO_CLI_ARG_MAP[key]: value
            for key, value in arguments.items()
            if key in _PREPRO_CLI_ARG_MAP
        }))

    run(arguments, sim_overrides=sim_overrides,
        prepro_overrides=prepro_overrides)


if __name__ == '__main__':
    main()
