#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

"""Load raw session's stimulus array from file."""

# pylint:disable=missing-docstring

from os.path import isfile, join

import numpy as np


def load_raw_stimulus(input_dir, filename):
    """Load the stimulus array for the session from filename.

    Args:
        - `input_dir` (str): `input_dir` simulation parameter (eg set from
            command line, during `__init__.run()` call or in `simulation`
            subtree).
        - `filename` (str): `filename` parameter from the `inputs` session parameter.

    Return:
        (np.ndarray): (nframes * nfilter * nrows * ncolumns) numpy array. The
            array is loaded:
                3- If os.path.join(`input_dir`, `filename`) points towards
                    a loadable numpy array, load it. Otherwise,
                4- If `input_dir` points to a formatted input directory and
                    `filename` is the name of a stimulus yaml filename load the
                    stimulus from yaml accordingly
            If none of the above works, an exception is thrown.
    """

    path = join(input_dir, filename)
    if not isfile(path):
        error_string = (
            f"Couldn't load an input stimulus with: \n`input_dir`"
            f"simulation parameter: {input_dir} and \n`filename`"
            f"session parameter: {filename}.\n Please check params."
        )
        raise FileNotFoundError(error_string)

    print(f"-> Loading input from array at os.path.join(`input_dir`,"
          f"`filename`): `{path}`")
    return np.load(path)
