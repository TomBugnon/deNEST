#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Spiking VisNet."""

import logging
import os

import time
from .parameters import Params
from .simulation import Simulation
from .network import Network
from .save import load_yaml

from .utils import misc

__all__ = ['load_params', 'run', 'Simulation', 'Network']

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'stdout': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'loggers': {
        'nets': {
            'level': 'INFO',
            'handlers': ['stdout'],
        }
    }
})
log = logging.getLogger(__name__)  # pylint: disable=invalid-name

SEPARATOR = ('\n'
             '==============================================================\n'
             '==============================================================\n'
             '==============================================================\n')


def load_params(path, *overrides):
    """Load a list of parameter files, optionally overriding some values.

    Args:
        path (str): The filepath to load.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.

    Returns:
        Params: The loaded parameters with overrides applied.
    """
    print(f'Loading parameters: `{path}`', end='', flush=True)
    if overrides:
        print(f' with {len(overrides)} override trees.', end='')
    print('...')
    path_dir = os.path.dirname(os.path.abspath(path))
    return Params.merge(
        *[Params(overrides_tree)
          for overrides_tree in overrides],
        *[Params.load(path_dir, relative_path)
        for relative_path in load_yaml(path)]
    )


def run(path, *overrides, output_dir=None, input_path=None):
    """Run the simulation described by the params at ``path``.

    Args:
        path (str): The filepath of a parameter file specifying the simulation.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.
    """
    start_time = time.time() # Timing of simulation time
    print(SEPARATOR)

    # Load parameters
    print('Load params...\n')
    params = load_params(path, *overrides)
    # Incorporate kwargs in params
    if output_dir is not None:
        print(f'Overriding output directory: {output_dir}')
        params.c['simulation']['output_dir'] = output_dir
    if input_path is not None:
        print(f'Overriding input: {input_path}')
        params.c['simulation']['input_path'] = input_path
    print('\n...done loading params.', flush=True, end=SEPARATOR)

    # Initialize simulation
    print('Initialize simulation...\n', flush=True)
    sim = Simulation(params)
    print('\n...done initializing simulation...', flush=True, end=SEPARATOR)

    # Save simulation metadata
    print('Save simulation metadata...\n', flush=True)
    sim.save_metadata()
    print('\n...done saving simulation metadata...', flush=True, end=SEPARATOR)

    # Simulate and save
    if not params.get(('simulation', 'dry_run'), False):
        print('Run simulation...\n', flush=True)
        sim.run()
        print('\n...done running simulation...', flush=True, end=SEPARATOR)

    # Save data after all sessions have been run.
    if params.get(('simulation', 'save_simulation'), True):
        print('Save simulation...\n', flush=True)
        sim.save_data()
        print('\n...done saving simulation...', flush=True, end=SEPARATOR)

    # Dump network's connections
    if params.get(('simulation', 'dump_connections'), False):
        print('Dump connections...\n', flush=True)
        sim.dump_connections()
        print('\n...done dumping connections...', flush=True, end=SEPARATOR)

    # Plot network's connections
    if params.get(('simulation', 'plot_connections'), False):
        print('Plot connections...\n', flush=True)
        sim.plot_connections()
        print('\n...done plotting connections...', flush=True, end=SEPARATOR)

    # Dump network's incoming connection numbers per layer
    if params.get(('simulation', 'dump_connection_numbers'), False):
        print('Dumping connection numbers...\n', flush=True)
        sim.dump_connection_numbers()
        print('\n...done dumping conn numbers...', flush=True, end=SEPARATOR)

    # Conclusive remarks
    print('\nThis simulation is a great success.\n')
    print(f"Total simulation virtual time: {sim.total_time()}ms")
    print(f"Total simulation real time: {misc.pretty_time(start_time)}")
    print('\nSimulation output can be found at the following path:')
    print(os.path.abspath(sim.output_dir), '\n')
