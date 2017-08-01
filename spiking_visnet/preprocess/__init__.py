#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/__init__.py

"""Preprocess all raw stimuli."""

from .. import load_params as _load_params
from .. import load_yaml as _load_yaml
from ..network import Network as _Network
from .preprocess import preprocess_all as _preprocess_all
from .utils import preprocessing_subdir as _preprocessing_subdir

INPUT_SUBDIRS = {'raw_input': 'raw_input',
                 'preprocessed_input': 'preprocessed_input',
                 'raw_input_sets': 'raw_input_sets',
                 'preprocessed_input_sets': 'preprocessed_input_sets',
                 'stimuli': 'stimuli'}


def run(args):
    """Run preprocessing."""
    # Load network
    params = _load_params(args['<network_params>'])
    network_params = params['children']['network']['children']
    network = _Network(network_params)

    # Load preprocessing parameters
    prepro_params = _load_yaml(args['<preprocessing_params>'])

    _preprocess_all(args['<input_dir>'],
                    _preprocessing_subdir(prepro_params, network),
                    network,
                    prepro_params)
