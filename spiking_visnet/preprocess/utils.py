#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/utils.py

"""Preprocessing utils."""

import os
import os.path

from . import filt, normalize, resize

NAME_MAPPING = {
    'resize': resize.get_string,
    'filter': filt.get_string,
    'normalize': normalize.get_string
}


def preprocessing_subdir(prepro_params, network):
    """Generate a string describing the preprocessing pipeline.

    Uses the get_string function in each of the preprocessing steps' modules
    to generate a string describing the preprocessing pipeline.
    """
    subdir = ''
    for prepro_step in prepro_params['preprocessing']:
        # use the get_string function in each of the preprocessing modules
        subdir += NAME_MAPPING[prepro_step](prepro_params, network)
    return subdir.strip('_')


def create_set(input_dir, setname, filenames):
    """Create a raw input set of symlinks to elements of filenames."""
    assert(isinstance(filenames, list))
    set_dir = os.path.join(input_dir, 'raw_input_sets', setname)
    os.makedirs(set_dir, exist_ok=True)
    # Create symlinks
    for filename in filenames:
        symlink_source = os.path.join('..', '..', 'raw_input', filename)
        symlink_target = os.path.join(set_dir, filename)
        os.symlink(symlink_source, symlink_target)
