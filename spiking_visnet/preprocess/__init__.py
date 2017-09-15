#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/__init__.py

"""Preprocess all raw stimuli."""

from .. import load_params as _load_params
from ..network import Network as _Network
from ..save import load_yaml as _load_yaml
from .preprocess import preprocess_all as _preprocess_all
from .utils import preprocessing_subdir as _preprocessing_subdir
from ..utils.structures import chaintree, dictify
from ..parameters import AutoDict


def run(args, sim_overrides=None, prepro_overrides=None):
    """Run preprocessing."""
    # Load network
    params_path = args['<sim_params>']
    full_params_tree = _load_params(params_path, overrides=sim_overrides)
    network_params = full_params_tree['children']['network']['children']
    sim_params = full_params_tree['children']['simulation']
    network = _Network(network_params, sim_params)

    # Load preprocessing parameters
    prepro_params = load_preprocessing_params(args['<preprocessing_params>'],
                                              overrides=prepro_overrides)

    _preprocess_all(args['--input'],
                    _preprocessing_subdir(prepro_params, network),
                    network,
                    prepro_params)

def load_preprocessing_params(path, overrides=None):
    """Load Params from yaml containing a tree.

    Difference from spiking_visnet.parameters.load_params() is that the
    simulation parameter path points to a yaml containing a list of files to
    merge, whereas the preprocessing parameters path points directly to the
    parameters yaml.

    """
    return dictify(AutoDict(chaintree([_load_yaml(path), overrides])))
