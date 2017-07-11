#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __main__.py

"""
Spiking VisNet
~~~~~~~~~~~~~~

Usage:
    spiking_visnet <param_file.yml>
    spiking_visnet -h | --help
    spiking_visnet -v | --version

Arguments:
    <param_file.yml>         File comtaining simulation parameters

Options:
    -h --help                  Show this
    -v --version               Show version
"""

import sys

from docopt import docopt

from . import run
from .__about__ import __version__


if __name__ == '__main__':
    # Get command-line args from docopt.
    sys.argv[0] = 'spiking_visnet'
    args = docopt(__doc__, version=__version__)
    run(args['<param_file.yml>'])
