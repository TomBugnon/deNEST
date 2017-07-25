#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

from os.path import abspath as _abspath
from os.path import dirname as _dirname
from os.path import join as _join

import yaml

from .network import Network
from .simulation import Simulation
from .utils.structures import chaintree as _chaintree


def load_yaml(*args):
    """Load yaml file from joined (os.path.join) arguments."""
    with open(_join(*args), 'rt') as f:
        return yaml.load(f)


def load_params(path):
    """Return the full parameter tree described by the file at path."""
    directory = _dirname(_abspath(path))
    params = [
        load_yaml(directory, relative_path)
        for relative_path in load_yaml(path)
    ]
    return _chaintree(params)


def init(params, path):
    """Initialize NEST network from the full parameter tree."""
    print('Initializing network...')
    # Get relevant parts of the full simulation tree
    network_params = params['children']['network']['children']
    kernel_params = params['children']['kernel']
    # Build the network object
    network = Network(network_params)
    # Get the saving subdirectory. sim outputs are in SAVE_DIR/save_subdir_str
    network.get_save_subdir_str(params, path)
    # Initialize kernel + network in NEST
    network.init_nest(kernel_params)
    # Get the bi-directional GID-location mappings for each population.
    network.get_gid_location_mappings()
    import ipdb
    ipdb.set_trace()

    print('...done initializing network.')
    return network


def simulate(params):
    """Simulate all sessions described in parameter tree."""
    simulation_params = params['children']['simulation']
    print(f'Simulating...', flush=True)
    Simulation(simulation_params).run()
    print('...finished simulation.', flush=True)


def run(path):
    """Run all."""
    print(f'Loading parameters: `{path}`... ', end='', flush=True)
    params = load_params(path)
    print('done.', flush=True)
    # Initialize kernel and network
    network = init(params, path)
    # Simulate
    simulate(params)
