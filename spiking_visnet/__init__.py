#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

import yaml
from os.path import abspath, dirname, join

from .nestify.format_net import get_network as _get_network
from .nestify.init_nest import init_network as _init_network
from .utils.structures import chaintree as _chaintree


def _load_yaml(*args):
    with open(join(*args), 'rt') as f:
        return yaml.load(f)


def load_params(path):
    directory = dirname(abspath(path))
    params = [
        _load_yaml(directory, relative_path)
        for relative_path in _load_yaml(path)
    ]
    return _chaintree(params)


def init(path):
    # Load parameters
    params = load_params(path)
    '''Initialize the network in the NEST kernel.'''
    # Get relevant parts of the full simulation tree
    network_params = params['children']['network']['children']
    kernel_params = params['children']['kernel']

    network = _get_network(network_params)
    _init_network(network, kernel_params)

    return network


def simulate(network):
    print('Simulating...', end='', flush=True)
    print('done.')


def run(path):
    print(f'Running: `{path}`...', flush=True)
    # Initialize network
    network = init(path)
    # Simulate
    simulate(network)
