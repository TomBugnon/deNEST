#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __main__.py

"""
Spiking VisNet
~~~~~~~~~~~~~~

Usage:
    python -m spiking_visnet <param_file.yml> [-i <input>] [-s <savedir>]
    python -m spiking_visnet -h | --help
    python -m spiking_visnet -v | --version

Arguments:
    <param_file.yml>  File containing simulation parameters

Options:
    -s <savedir> --savedir <savedir>    Directory in which simulation results will be saved. Overwrites config file default if specified.
    -i <input> --input <input>    Path to a stimulus np-array to show to the network during all sessions. Overwrites session parameter's 'session_stims' if specified.
    -h --help                           Show this
    -v --version                        Show version
"""

import random
import sys

from config import PYTHON_SEED
from docopt import docopt

from . import run
from .__about__ import __version__

random.seed(PYTHON_SEED)

if __name__ == '__main__':
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m spiking_visnet` calling pattern.
    argv = ['-m', 'spiking_visnet'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv, version=__version__)
    # Run it!
    run(arguments['<param_file.yml>'], cli_args=arguments)
