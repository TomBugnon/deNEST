#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __main__.py

"""
Spiking VisNet
~~~~~~~~~~~~~~

Usage:
    python -m spiking_visnet <param_file.yml>
    python -m spiking_visnet -h | --help
    python -m spiking_visnet -v | --version

Arguments:
    <param_file.yml>  File containing simulation parameters

Options:
    -h --help         Show this
    -v --version      Show version
"""

import sys

from docopt import docopt

from . import run
from .__about__ import __version__

if __name__ == '__main__':
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m spiking_visnet` calling pattern.
    argv = ['-m', 'spiking_visnet'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv, version=__version__)
    # Run it!
    run(arguments['<param_file.yml>'])
