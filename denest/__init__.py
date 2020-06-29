#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""deNEST: a declarative frontend for NEST"""

import logging.config
import time
from pathlib import Path
from pprint import pformat

from .__about__ import *
from .io.load import load_yaml
from .network import Network
from .parameters import ParamsTree
from .session import Session
from .simulation import Simulation
from .utils import misc

__all__ = [
    "load_trees", "run", "Simulation", "Network", "Session", "ParamsTree"
]

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s [%(name)s] %(levelname)s: %(message)s"}
        },
        "handlers": {
            "stdout": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "loggers": {"denest": {"level": "INFO", "handlers": ["stdout"],}},
    }
)
log = logging.getLogger(__name__)


def load_trees(path, *overrides):
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
    log.info("Loading parameter file paths from %s", str(path))
    if overrides:
        log.info("Using %s override tree(s)", len(overrides))
    rel_path_list = load_yaml(path)
    log.info("Finished loading parameter file paths")
    log.info("Loading parameters files: \n%s", pformat(rel_path_list))
    return ParamsTree.merge(
        *[ParamsTree(overrides_tree) for overrides_tree in overrides],
        *[
            ParamsTree.read(Path(path.parent, relative_path))
            for relative_path in rel_path_list
        ],
        name=path,
    )
    log.info("Finished loading parameter files.")


def run(path, *overrides, output_dir=None, input_dir=None):
    """Run the simulation specified by the parameters at ``path``.

    Args:
        path (str): The filepath of a parameter file specifying the simulation.
        *overrides (tree-like): Variable number of tree-like parameters that
            should override those from the path. Last in list is applied first.

    Keyword Args:
        input_dir (str | None): ``None`` or the path to the input. Passed to
            :class:`Simulation`. If defined, overrides the ``input_dir``
            simulation parameter.
        output_dir (str | None): None or the path to the output directory.
            Passed to :class:`Simulation` If defined, overrides the
            ``output_dir`` simulation parameter.
    """
    # Timing of simulation time
    start_time = time.time()
    log.info(
        "\n\n=== RUNNING SIMULATION ========================================================\n"
    )

    # Load parameters
    tree = load_trees(path, *overrides)

    # Initialize simulation
    log.info("Initializing simulation...")
    sim = Simulation(tree, input_dir=input_dir, output_dir=output_dir)
    log.info("Finished initializing simulation")

    # Simulate
    log.info("Running simulation...")
    sim.run()
    log.info("Finished running simulation")

    # Final logging
    log.info("Total simulation virtual time: %s ms", sim.total_time())
    log.info("Total simulation real time: %s", misc.pretty_time(start_time))
    log.info("Simulation output written to: %s", Path(sim.output_dir).resolve())
