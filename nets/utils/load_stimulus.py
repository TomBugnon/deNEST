#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

"""Load raw session's stimulus array from file."""

# pylint:disable=missing-docstring

from os.path import isfile, join

from ..save import load_as_numpy


def load_raw_stimulus(input_path, session_input):
    """Load the stimulus array for the session from file.

    Args:
        - `input_path` (str): `input_path` simulation parameter (eg set from
            command line, during `__init__.run()` call or in `simulation`
            subtree).
        - `session_input` (str): `session_input` session parameter.

    Return:
        (np.ndarray): (nframes * nfilter * nrows * ncolumns) numpy array. The
            array is loaded:
                1- If `input_path` points towards a loadable array, ignore the
                    `session_input` parameter and load it. Otherwise,
                2- If `session_input` points towards a loadable array, load it.
                    Otherwise,
                3- If os.path.join(`input_path`, `session_input`) points towards
                    a loadable numpy array, load it. Otherwise,
                4- If `input_path` points to a formatted input directory and
                    `session_input` is the name of a stimulus yaml file load the
                    stimulus from yaml accordingly
            If none of the above works, an exception is thrown.
    """

    # Option 1: load directly from input_path
    if isfile(input_path):
        try:
            movie = load_as_numpy(input_path)
            print(f"-> Loading input from array at simulation's `input_path`:"
                  f'`{input_path}` (loading option 1)')
            return movie
        except FileNotFoundError:
            pass
    # Option 2: load directly from session_input
    if isfile(session_input):
        try:
            movie = load_as_numpy(session_input)
            print(f"-> Loading input from array at session's `session_input`:"
                  f'`{session_input}` (loading option 2)')
            return movie
        except FileNotFoundError:
            pass
    fullpath = join(input_path, session_input)
    # Option 3: load from os.path.join(input_path, session_input)
    if isfile(fullpath):
        try:
            movie = load_as_numpy(fullpath)
            print(f"-> Loading input from array at os.path.join(`input_path`,"
                  f"`session_input`): `{fullpath}` (loading option 3)")
            return movie
        except FileNotFoundError:
            pass
    error_string = (f"Couldn't load an input stimulus with: \n`input_path`"
                    f"simulation parameter: {input_path} and \n`session_input`"
                    f"session parameter: {session_input}")
    raise Exception(error_string)
