#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/__init__.py

"""Preprocess all raw stimuli."""

from .. import load_params as _load_params
from ..network import Network as _Network
from ..save import load_yaml as _load_yaml
from .preprocess import preprocess_all as _preprocess_all
from .utils import preprocessing_subdir as _preprocessing_subdir


def run(args):
    """Run preprocessing."""
    # Load network
    params_path = args['<sim_params>']
    full_params_tree = _load_params(params_path)
    network_params = full_params_tree['children']['network']['children']
    network = _Network(network_params, params_path)

    # Load preprocessing parameters
    prepro_params = _load_yaml(args['<preprocessing_params>'])

    _preprocess_all(args['<input_dir>'],
                    _preprocessing_subdir(prepro_params, network),
                    network,
                    prepro_params)
