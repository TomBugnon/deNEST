#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/__main__.py


"""
preprocessing
~~~~~~~~~~~~~~

Usage:
    python -m spiking_visnet.preprocess -i <input_dir> -p <preprocessing_params> -n <network_params>
    python -m spiking_visnet.preprocess -h | --help

Arguments:
    <param_file.yml>  File comtaining simulation parameters

Options:
    -h --help         Show this
"""

import sys

from docopt import docopt

from . import run

if __name__ == '__main__':
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m spiking_visnet` calling pattern.
    argv = ['-m', 'spiking_visnet.preprocess'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv)
    run(arguments)
