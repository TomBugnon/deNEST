#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/__main__.py


"""
preprocessing.

~~~~~~~~~~~~~~

Usage:
    python -m spiking_visnet.preprocess -p <preprocessing_params> -n <sim_params> [-i <input_dir>]
    python -m spiking_visnet.preprocess -h | --help

Arguments:
    <preprocessing_params>  Relative path to preprocessing parameter yaml file
    <sim_params>            Relative path to full simulation parameter file

Options:
    -i --input=<input_dir>  Input directory. If not specified, uses INPUT_DIR from config.
    -h --help               Show this
"""

import random
import sys

from config import PYTHON_SEED
from docopt import docopt
from user_config import INPUT_DIR

from . import run

random.seed(PYTHON_SEED)

if __name__ == '__main__':
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m spiking_visnet` calling pattern.
    argv = ['-m', 'spiking_visnet.preprocess'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv)
    if not arguments['--input']:
        arguments['--input'] = INPUT_DIR

    run(arguments)
