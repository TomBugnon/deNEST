#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

"""Load raw session's stimulus array from file."""


from pathlib import Path

import numpy as np


def load_raw_stimulus(input_dir, filename):
    """Load the stimulus array for the session from filename.

    Args:
        input_dir (str): ``input_dir`` simulation parameter (e.g. set from
            the command line, during an ``__init__.run()`` call, or in the
            ``simulation`` parameter subtree).
        filename (str): ``filename`` parameter from the ``inputs`` session
            parameter.

    Return:
        (np.ndarray): ``(nframes * nfilter * nrows * ncolumns)`` numpy array.
            1. If ``<input_dir>/<filename>`` points towards a loadable
               numpy array, it is loaded and returned. Otherwise,
            2. If ``input_dir`` points to a formatted input directory and
               ``filename`` is the name of a stimulus file, load the stimulus
               from the file accordingly.

    Raises:
        FileNotFoundError: If no array was found.
    """
    path = Path(input_dir, filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Couldn't load an input stimulus with: \n`input_dir` "
            f"simulation parameter: {input_dir} and \n`filename` "
            f"session parameter: {filename}.\nPlease check params."
        )
    print(f"-> Loading input from array at {path}")
    return np.load(path)
