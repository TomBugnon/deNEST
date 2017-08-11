#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Spiking VisNet."""

from .network import Network
from .parameters import Params, load_params
from .save import load_yaml, save_all
from .simulation import Simulation


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


def simulate(network, params):
    """Simulate a network.

    Args:
        network (Network): The network to simulate.
        params (dict-like): The simulation parameters.
    """
    print(f'Simulating...', flush=True)
    simulation = Simulation(params['children']['sessions'])
    simulation.run(params, network)
    print('...finished simulation.', flush=True)
    save_all(network, simulation, params)


def run(path, overrides=None):
    """Run the simulation described by the params at ``path``.

    Args:
        path (str): The filepath of a parameter file specifying the simulation.

    Keyword Arguments:
        overrides (dict-like): Any parameters that should override those from
            the path.
    """
    print(f'Loading parameters: `{path}`... ', end='', flush=True)
    params = load_params(path, overrides=overrides)
    print('done.', flush=True)
    # Initialize kernel and network
    network = init(params)
    # Simulate and save.
    simulate(network, params)
