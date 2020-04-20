#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Spiking VisNet."""

import logging
import os

import time
from .parameters import Tree
from .simulation import Simulation
from .network import Network
from .io.load import load_yaml

from .utils import misc

__all__ = ['load_tree', 'run', 'Simulation', 'Network']

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


def load_tree(path, *overrides):
    """Load a list of parameter files, optionally overriding some values.

    Args:
        path (str): The filepath to load.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.

    Returns:
        Tree: The loaded parameter tree with overrides applied.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No parameter file at {path}")
    print(f'Loading parameters from list at: `{path}`', flush=True)
    if overrides:
        print(f' with {len(overrides)} override trees.', end='')
    path_dir = os.path.dirname(os.path.abspath(path))
    rel_path_list = load_yaml(path)
    print(f"List of loaded parameter files: {rel_path_list}", flush=True)
    return Tree.merge(
        *[Tree(overrides_tree)
          for overrides_tree in overrides],
        *[Tree.read(os.path.join(path_dir, relative_path))
          for relative_path in rel_path_list]
    )


def run(path, *overrides, output_dir=None, input_path=None):
    """Run the simulation described by the parameter tree at ``path``.

    Args:
        path (str): The filepath of a parameter file specifying the simulation.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.

    Kwargs:
        input_path (str | None): None or the path to the input. Passed to
            ``Simulation.__init__``. If defined, overrides the `input_path`
            simulation parameter
        output_dir (str | None): None or the path to the output directory.
            Passed to ``Simulation.__init__`` If defined, overrides `output_dir`
            simulation parameter.
    """
    start_time = time.time()  # Timing of simulation time
    print(SEPARATOR)

    # Load parameters
    print('Load parameter tree...\n')
    tree = load_tree(path, *overrides)
    print('\n...done loading parameter tree.', flush=True, end=SEPARATOR)

    # Initialize simulation
    print('Initialize simulation...\n', flush=True)
    sim = Simulation(tree, input_path=input_path, output_dir=output_dir)
    print('\n...done initializing simulation...', flush=True, end=SEPARATOR)

    # Simulate
    print('Run simulation...\n', flush=True)
    sim.run()
    print('\n...done running simulation...', flush=True, end=SEPARATOR)

    # Conclusive remarks
    print('\nThis simulation is a great success.\n')
    print(f"Total simulation virtual time: {sim.total_time()}ms")
    print(f"Total simulation real time: {misc.pretty_time(start_time)}")
    print('\nSimulation output can be found at the following path:')
    print(os.path.abspath(sim.output_dir), '\n')
