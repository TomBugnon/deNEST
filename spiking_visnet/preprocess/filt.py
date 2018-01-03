#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filt.py
"""Filter (nframes * nrows * ncols) to (nframes * nfilters * nrows * ncols)."""

import numpy as np

from ..utils.filter_suffixes import get_summary_string

# TODO


def filter_movie(input_movie, preprocessing_params, network):
    """Filter input_movie with set of filters described in network.

    Args:
        input_movie (np.array): (nframes * nrows * ncols) np-array
            preprocessing_params (dict)
        network (Network object)

    Returns:
        np.array: (nframes * nfilters * nrows * ncols) np.array.

    """
    nframes, nrows, ncols = np.shape(input_movie)
    return np.reshape(input_movie, (nframes, 1, nrows, ncols))


# TODO
def get_string(_, network):
    """Return summary string of this preprocessing step."""
    return 'filter' + get_summary_string(network.filters())
