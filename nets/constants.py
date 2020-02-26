#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py
"""Package-wide constants."""

DEFAULT_PARAMS_PATH = None

DEFAULT_INPUT_PATH = './input'

INPUT_SUBDIRS = {  # Ignored if input stimuli are loaded from .npy
    'raw_input': 'raw_input',
    'preprocessed_input': 'preprocessed_input',
    'raw_input_sets': 'raw_input_sets',
    'preprocessed_input_sets': 'preprocessed_input_sets',
    'stimuli': 'stimuli'
}

DEFAULT_OUTPUT_DIR = './output'

METADATA_FILENAME = 'metadata.yml'

PYTHON_SEED = 94

NEST_SEED = 94
