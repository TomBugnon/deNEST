#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Spiking VisNet."""

from pathlib import Path
import logging

import time
from .parameters import ParamsTree
from .simulation import Simulation
from .network import Network
from .io.load import load_yaml

from .utils import misc

__all__ = ['load_paramstree', 'run', 'Simulation', 'Network']

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


def load_paramstree(path, *overrides):
    """Load a list of parameter files, optionally overriding some values.

    Args:
        path (str): The filepath to load.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.

    Returns:
        ParamsTree: The loaded parameter tree with overrides applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No parameter file at {path}")
    print(f'Loading parameters from list at: `{path}`', flush=True)
    if overrides:
        print(f' with {len(overrides)} override trees.', end='')
    rel_path_list = load_yaml(path)
    print(f"List of loaded parameter files: {rel_path_list}", flush=True)
    return ParamsTree.merge(
        *[ParamsTree(overrides_tree)
          for overrides_tree in overrides],
        *[ParamsTree.read(Path(path.parent, relative_path))
          for relative_path in rel_path_list],
        name=path,
    )


def run(path, *overrides, output_dir=None, input_dir=None):
    """Run the simulation described by the parameter tree at ``path``.

    Args:
        path (str): The filepath of a parameter file specifying the simulation.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.

    Kwargs:
        input_dir (str | None): None or the path to the input. Passed to
            ``Simulation.__init__``. If defined, overrides the `input_dir`
            simulation parameter
        output_dir (str | None): None or the path to the output directory.
            Passed to ``Simulation.__init__`` If defined, overrides `output_dir`
            simulation parameter.
    """
    start_time = time.time()  # Timing of simulation time
    print(SEPARATOR)

    # Load parameters
    print('Load parameter tree...\n')
    tree = load_paramstree(path, *overrides)
    print('\n...done loading parameter tree.', flush=True, end=SEPARATOR)

    # Initialize simulation
    print('Initialize simulation...\n', flush=True)
    sim = Simulation(tree, input_dir=input_dir, output_dir=output_dir)
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
    print(Path(sim.output_dir).resolve(), '\n')
