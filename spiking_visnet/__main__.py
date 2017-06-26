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

__version__ = '0.0.0'

from os.path import join, dirname, abspath
import pprint
import sys

from docopt import docopt
import yaml

from .utils.structures import chaintree
from .nestify.format_net import get_Network
from .nestify.init_nest import init_Network


def load_yaml(*args):
    with open(join(*args), 'rt') as f:
        return yaml.load(f)


def load_params(path):
    directory = dirname(abspath(path))
    params = [
        load_yaml(directory, relative_path)
        for relative_path in load_yaml(path)
    ]
    return chaintree(params)


def main(args):
    params = load_params(args['<param_file.yml>'])

    # Get relevant parts of the full simulation tree
    net_tree = params['children']['network']['children']
    kernel_params = params['children']['kernel']

    net = get_Network(net_tree)
    net, kernel_init = init_Network(net, kernel_params)
    return


if __name__ == '__main__':
    # Get command-line args from docopt.
    sys.argv[0] = 'spiking_visnet'
    args = docopt(__doc__, version=__version__)
    main(args)
