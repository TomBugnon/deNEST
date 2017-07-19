#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

import yaml
from os.path import abspath, dirname, join

from .network import Network

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
    import nest
    from .nestify.init_nest import init_network

    # Load parameters
    params = load_params(path)
    # Get relevant parts of the full simulation tree
    network_params = params['children']['network']['children']
    kernel_params = params['children']['kernel']

    # Build the network object
    network = Network(network_params)

    # Initialize the network in the NEST kernel
    init_network(network, kernel_params)

    return network


# TODO: finish
def simulate(t):
    import nest
    print(f'Simulating for {t} ms... ', flush=True)
    nest.Simulate(t)
    print('...done.')


# TODO: finish
def run(path, t):
    print(f'Running: `{path}`...', flush=True)
    # Initialize network
    network = init(path)
    # Simulate
    simulate(t)
