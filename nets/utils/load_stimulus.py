#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

"""Load raw session's stimulus from file."""

# pylint:disable=missing-docstring

from os.path import basename, isfile, join

from ..save import load_as_numpy


def load_raw_stimulus(input_path, session_input):
    """Load the stimulus for the session from `input_path` and `session_input`.

    Args:
        - `input_path` (str): `input_path` simulation parameter (eg set from
            command line, during `__init__.run()` call or in `simulation`
            subtree).
        - `session_input` (str): `session_input` session parameter.

    Return:
        (np.ndarray): (time * filter * rows * columns) numpy array. The array is
            loaded or created as follows:
                1- If `input_path` points towards a loadable array, ignore the
                    `session_input` parameter and load it. Otherwise,
                2- If `session_input` points towards a loadable array, load it.
                    Otherwise,
                3- If os.path.join(`input_path`, `session_input`) points towards
                    a loadable numpy array, load it. Otherwise,
                4- If `input_path` points to a formatted input directory and
                    `session_input` is the name of a stimulus yaml file load the
                    stimulus from yaml accordingly
            If none of the above works, we throw an exception.

    Return:
    (tuple): (<stim>, <filenames>, <stim_metadata> ) where:
        <stim > (np - array): (nframes * nfilters * nrows * ncols) array.
        <frame_filenames> (list): list of length nframes containing the
            filename of the movie each frame is taken from.
        <stim_metadata > (dict or None):
            None if stimulus is loaded directly from a numpy array.
            Metadata of the preprocessing pipeline (used to map input layers
            and filter dimensions) otherwise.
    """

    # Option 1: load directly from input_path
    if isfile(input_path):
        try:
            movie = load_as_numpy(input_path)
            print(f"-> Loading input from array at simulation's `input_path`:"
                  f'`{input_path}` (loading option 1)')
            return (movie,
                    frame_labels_from_file(input_path, len(movie)),
                    None)
        except FileNotFoundError:
            pass
    # Option 2: load directly from session_input
    if isfile(session_input):
        try:
            movie = load_as_numpy(session_input)
            print(f"-> Loading input from array at session's `session_input`:"
                  f'`{session_input}` (loading option 2)')
            return (movie,
                    frame_labels_from_file(input_path, len(movie)),
                    None)
        except FileNotFoundError:
            pass
    fullpath = join(input_path, session_input)
    # Option 3: load from os.path.join(input_path, session_input)
    if isfile(fullpath):
        try:
            movie = load_as_numpy(fullpath)
            print(f"-> Loading input from array at os.path.join(`input_path`,"
                  f"`session_input`): `{fullpath}` (loading option 3)")
            return (movie,
                    frame_labels_from_file(input_path, len(movie)),
                    None)
        except FileNotFoundError:
            pass
    error_string = (f"Couldn't load an input stimulus with: \n`input_path`"
                    f"simulation parameter: {input_path} and \n`session_input`"
                    f"session parameter: {session_input}")
    raise Exception(error_string)


def frame_labels_from_file(input_path, nframes):
    # If the input is loaded from numpy, all the frames have the same label,
    # which is the filename.
    return [basename(input_path)
            for i in range(nframes)]
