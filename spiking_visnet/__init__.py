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
from .params import Params


def load_params(path, overrides=None):
    """Load a parameter file, optionally overriding some values.

    Args:
        path (str): The filepath to load.

    Keyword Args:
        overrides (dict): A dictionary containing parameters that will take
            precedence over those in the file.

    Returns:
        Params: the loaded parameters with overrides applied.
    """
    directory = _dirname(_abspath(path))
    trees = [load_yaml(directory, relative_path)
             for relative_path in load_yaml(path)]
    if overrides:
        trees.append(overrides)
    return Params(_chaintree(trees))


def incorporate_user_args(params_tree, user_input=None, user_savedir=None):
    """Incorporate user cli arguments at proper locations in full params tree.
    """
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


def run(path, overrides=None):
    """Run all."""
    print(f'Loading parameters: `{path}`... ', end='', flush=True)
    params = load_params(path, overrides=overrides)
    print('done.', flush=True)
    # Initialize kernel and network
    network = init(params)
    # Simulate
    simulate(network, params)
    # Save
    save_all(network, params)
