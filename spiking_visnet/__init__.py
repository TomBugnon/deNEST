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


def load_yaml(*args):
    with open(_join(*args), 'rt') as f:
        return yaml.load(f)


def load_params(path):
    # import ipdb; ipdb.set_trace()
    directory = _dirname(_abspath(path))
    params = [
        load_yaml(directory, relative_path)
        for relative_path in load_yaml(path)
    ]
    return _chaintree(params)


def init(params):
    print('Initializing network...')
    from .nestify.init_nest import init_network
    # Get relevant parts of the full simulation tree
    network_params = params['children']['network']['children']
    kernel_params = params['children']['kernel']
    # Build the network object
    network = Network(network_params)
    # Initialize the network in the NEST kernel
    init_network(network, kernel_params)
    print('...done initializing network.')
    return network


# TODO: define Session and Simulation classes
def simulate(params):
    simulation_params = params['children']['simulation']
    print(f'Simulating...', flush=True)
    Simulation(simulation_params).run()
    print('...finished simulation.', flush=True)


def run(path):
    print(f'Loading parameters: `{path}`... ', end='', flush=True)
    params = load_params(path)
    print('done.', flush=True)
    # Initialize network
    network = init(params)
    # Simulate
    simulate(params)
