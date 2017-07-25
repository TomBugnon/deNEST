#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

from os.path import join, splitext

import nest

from ..user_config import SAVE_DIR
from .utils.system import mkdir_ifnot





def generate_save_subdir_str(full_params_tree, param_file_path):
    """Create and return relative path to the simulation saving directory.

    Returns:
        (str): The full path to the simulation saving directory  will be
            SAVE_DIR/subdir_str

    """
    param_file = splitext(param_file_path)[0]
    subdir_str = param_file
    return subdir_str
