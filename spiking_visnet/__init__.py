#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Run full simulation."""

from os.path import abspath as _abspath
from os.path import dirname as _dirname

from .network import Network
from .save import load_yaml, save_all
from .simulation import Simulation
from .utils.structures import chaintree as _chaintree


def load_params(path, cli_args={}):
    """Return the full parameter tree described by the file at path.

    - Add the path to the full params tree to the simulation parameters.
    - Possibly incorporate in the tree the command line arguments provided by
    user."""
    directory = _dirname(_abspath(path))
    params_tree = _chaintree([
        load_yaml(directory, relative_path)
        for relative_path in load_yaml(path)
    ])

    # Add path to sim params
    params_tree['children']['simulation']['param_file_path'] = path

    return incorporate_user_args(params_tree,
                                 user_input=cli_args.get('--input', None),
                                 user_savedir=cli_args.get('--savedir', None))


def incorporate_user_args(params_tree, user_input=None, user_savedir=None):
    """Incorporate user cli arguments at proper locations in full params tree.
    """
    if user_input:
        (params_tree['children']
         ['sessions']['params']['user_input']) = user_input
    if user_savedir:
        (params_tree['children']
         ['simulation']['user_savedir']) = user_savedir
    return params_tree


def init(params):
    """Initialize NEST network from the full parameter tree."""
    print('Initializing network...')
    # Get relevant parts of the full simulation tree
    network_params = params['children']['network']['children']
    kernel_params = params['children']['kernel']
    sim_params = params['children']['simulation']
    # Build the network object
    network = Network(network_params, sim_params)
    # Initialize kernel + network in NEST
    network.init_nest(kernel_params, sim_params)
    print('...done initializing network.')
    return network


def simulate(network, params, user_input=None):
    """Simulate all sessions described in parameter tree."""
    print(f'Simulating...', flush=True)
    simulation = Simulation(params['children']['sessions'])
    simulation.run(params, network)
    print('...finished simulation.', flush=True)


def run(path, cli_args=None):
    """Run all."""
    print(f'Loading parameters: `{path}`... ', end='', flush=True)
    params = load_params(path, cli_args=cli_args)
    print('done.', flush=True)
    # Initialize kernel and network
    network = init(params)
    # Simulate
    simulate(network, params)
    # Save
    save_all(network, params)
