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
from .user_config import USER_OVERRIDES, DEFAULT_PARAMS_PATH

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
        'spiking_visnet': {
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


def run(path, *overrides, output_dir=None, input_dir=None):
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
    params = load_params(path, *overrides, USER_OVERRIDES)
    if DEFAULT_PARAMS_PATH is not None:
        default_params = load_params(DEFAULT_PARAMS_PATH)
        print('Merging default and simulation params...')
        params = Params.merge(params, default_params)
    # Incorporate kwargs in params
    if output_dir is not None:
        print(f'Overriding output directory: {output_dir}')
        params.c['simulation']['output_dir'] = output_dir
    if input_dir is not None:
        print(f'Overriding input: {input_dir}')
        params.c['simulation']['input_dir'] = input_dir
    print('\n...done loading params.', flush=True, end=SEPARATOR)

    # Initialize simulation
    print('Initialize simulation...\n', flush=True)
    sim = Simulation(params)
    print('\n...done initializing simulation...', flush=True, end=SEPARATOR)

    # Simulate
    if not params.c['simulation'].get('dry_run', False):
        print('Run simulation...\n', flush=True)
        sim.run()
        print('\n...done running simulation...', flush=True, end=SEPARATOR)

    # Save simulation
    if params.c['simulation'].get('save_simulation', True):
        print('Save simulation...\n', flush=True)
        sim.save()
        print('\n...done saving simulation...', flush=True, end=SEPARATOR)

    # Dump network's connections
    if params.c['simulation'].get('dump_connections', False):
        print('Dump connections...\n', flush=True)
        sim.dump_connections()
        print('\n...done dumping connections...', flush=True, end=SEPARATOR)

    # Plot network's connections
    if params.c['simulation'].get('plot_connections', False):
        print('Plot connections...\n', flush=True)
        sim.plot_connections()
        print('\n...done plotting connections...', flush=True, end=SEPARATOR)

    # Dump network's incoming connection numbers per layer
    if params.c['simulation'].get('dump_connection_numbers', False):
        sim.dump_connection_numbers()

    # Drop git hash
    misc.drop_git_hash(sim.output_dir)

    # Conclusive remarks
    print('\nThis simulation is a great success.\n')
    print(f"Total simulation virtual time: {sim.total_time()}ms")
    print(f"Total simulation real time: {misc.pretty_time(start_time)}")
    print('\nSimulation output can be found at the following path:')
    print(os.path.abspath(sim.output_dir), '\n')
