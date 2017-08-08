#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""
Spiking VisNet
"""

from .network import Network
from .save import load_yaml, save_all
from .simulation import Simulation
from .parameters import Params, load_params


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
