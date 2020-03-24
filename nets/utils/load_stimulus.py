#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

"""Load raw session's stimulus array from file."""

# pylint:disable=missing-docstring

from os.path import isfile, join

import numpy as np


def load_raw_stimulus(input_path, file):
    """Load the stimulus array for the session from file.

    Args:
        - `input_path` (str): `input_path` simulation parameter (eg set from
            command line, during `__init__.run()` call or in `simulation`
            subtree).
        - `file` (str): `file` parameter from the `inputs` session parameter.

    Return:
        (np.ndarray): (nframes * nfilter * nrows * ncolumns) numpy array. The
            array is loaded:
                1- If `input_path` points towards a loadable array, ignore the
                    `file` parameter and load it. Otherwise,
                2- If `file` points towards a loadable array, load it.
                    Otherwise,
                3- If os.path.join(`input_path`, `file`) points towards
                    a loadable numpy array, load it. Otherwise,
                4- If `input_path` points to a formatted input directory and
                    `file` is the name of a stimulus yaml file load the
                    stimulus from yaml accordingly
            If none of the above works, an exception is thrown.
    """

    # Option 1: load directly from `input_path`
    if isfile(input_path):
        path = input_path
        print(f"-> Loading input from array at simulation's `input_path`:"
              f'`{input_path}` (loading option 1)')
    # Option 2: load directly from `file`
    elif isfile(file):
        path = file
        print(f"-> Loading input from array at session's `file`:"
              f'`{file}` (loading option 2)')
    # Option 3: load from `os.path.join(input_path, file)`
    elif isfile(join(input_path, file)):
        path = join(input_path, file)
        print(f"-> Loading input from array at os.path.join(`input_path`,"
              f"`file`): `{path}` (loading option 3)")
    else:
        error_string = (
            f"Couldn't load an input stimulus with: \n`input_path`"
            f"simulation parameter: {input_path} and \n`file`"
            f"session parameter: {file}.\n Please check params."
        )
        raise FileNotFoundError(error_string)

    return np.load(path)
