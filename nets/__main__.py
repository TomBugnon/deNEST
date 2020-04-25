#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __main__.py
"""
NETS
~~~~

Usage:
    python -m nets <tree_paths.yml> [options]
    python -m nets -h | --help
    python -m nets -v | --version

Arguments:
    <tree_paths.yml>  YAML file containing list of relative paths of files to
                      load and merge into a parameter tree

Options:
    -o --output=PATH  Directory in which simulation results will be saved.
                      Overrides ``'output_dir'`` simulation parameter.
    -i --input=PATH   Path to a stimuli array to present to the network during
                      each session. Overrides <tree_paths.yml>.
    -h --help         Show this.
    -v --version      Show version.
"""

import sys

from docopt import docopt

from . import run
from .__about__ import __version__
from .utils.autodict import AutoDict

# Maps CLI options to their corresponding path in the parameter tree.
_CLI_ARG_MAP = {
    '<tree_paths.yml>': ('simulation', 'params', 'param_file_path'),
    '--input': ('simulation', 'params', 'input_dir'),
    '--output': ('simulation', 'params', 'output_dir'),
}


def main():
    """Run Spiking VisNet from the command line."""
    # Construct a new argument list to allow docopt's parser to work with the
    # `python -m nets` calling pattern.
    argv = ['-m', 'nets'] + sys.argv[1:]
    # Get command-line args from docopt.
    arguments = docopt(__doc__, argv=argv, version=__version__)
    # Get parameter overrides from the CLI options.
    overrides = AutoDict({
        _CLI_ARG_MAP[key]: value
        for key, value in arguments.items()
        if (value is not None and key in _CLI_ARG_MAP)
    })
    # Run it!
    run(arguments['<tree_paths.yml>'], overrides)


if __name__ == '__main__':
    main()
