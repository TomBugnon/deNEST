#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Spiking VisNet."""

import logging
import os

from .parameters import Params
from .simulation import Simulation
from .network.network import Network
from .save import load_yaml


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


def load_params(path, overrides=None):
    """Load a list of parameter files, optionally overriding some values.

    Args:
        path (str): The filepath to load.

    Keyword Args:
        overrides (dict): A dictionary containing parameters that will take
            precedence over those in the file.

    Returns:
        Params: The loaded parameters with overrides applied.
    """
    directory = os.path.dirname(os.path.abspath(path))
    return Params.merge(Params(overrides), *[
        Params.load(directory, relative_path)
        for relative_path in load_yaml(path)
    ])


def run(path, overrides=None, output_dir=None, input_dir=None):
    """Run the simulation described by the params at ``path``.

    Args:
        path (str): The filepath of a parameter file specifying the simulation.

    Keyword Arguments:
        overrides (dict-like): Any parameters that should override those from
            the path.
    """
    print(f'Loading parameters: `{path}`... ', end='', flush=True)
    params = load_params(path, overrides=overrides)
    # Incorporate kwargs in params
    if output_dir is not None:
        params.c['simulation']['output_dir'] = output_dir
    if output_dir is not None:
        params.c['simulation']['input_dir'] = input_dir
    print('done.', flush=True)
    # Initialize simulation
    sim = Simulation(params)
    # Simulate and save
    if not params.c['simulation']['dry_run']:
        sim.run()
    if params.c['simulation']['save_simulation']:
        sim.save()
    # Dump network's connections
    if params.c['simulation']['dump_connections']:
        sim.dump_connections()
    # Plot network's connections
    if params.c['simulation']['plot_connections']:
        sim.plot_connections()
