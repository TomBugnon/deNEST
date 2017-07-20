#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

import yaml
from os.path import abspath as _abspath
from os.path import dirname as _dirname
from os.path import join as _join

from .network import Network
from .simulation import Simulation

from .utils.structures import chaintree as _chaintree


def _load_yaml(*args):
    with open(_join(*args), 'rt') as f:
        return yaml.load(f)


def load_params(path):
    directory = _dirname(_abspath(path))
    params = [
        _load_yaml(directory, relative_path)
        for relative_path in _load_yaml(path)
    ]
    return _chaintree(params)


def init(params):
    from .nestify.init_nest import init_network

    # Get relevant parts of the full simulation tree
    network_params = params['children']['network']['children']
    kernel_params = params['children']['kernel']

    # Build the network object
    network = Network(network_params)

    # Initialize the network in the NEST kernel
    init_network(network, kernel_params)

    return network


# TODO: define Session and Simulation classes
def simulate(params):
    simulation = Simulation(params)
    print(f'Simulating', flush=True)
    simulation.run()
    print('...done simulating.')


def run(path):
    print(f'Running: `{path}`...', flush=True)
    # Load parameters
    params = load_params(path)
    # Initialize network
    network = init(params)
    # Simulate
    simulate(params)
    print('...All done.')
