#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""Spiking VisNet."""

import os

from .parameters import Params
from .simulation import Simulation
from .nestify.build import Network
from .save import load_yaml

__all__ = ['load_params', 'run', 'Simulation', 'Network']


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
    # Initialize simulation
    sim = Simulation(params)
    # Simulate and save.
    sim.run()
