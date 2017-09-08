#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sparsify.py

"""Save and load binary arrays sparsely."""

import numpy as np


# TODO: rename to save_sparse
def save_as_sparse(path, array):
    """Save a binary array in a sparse format.

    .. note::
        This assumes the array contains only binary values.
    """
    indices = np.where(array == 1)
    np.savez(path, indices)


# TODO: Rename to load_sparse
# TODO: implement another function that accepts either dense or sparse
def load_as_numpy(path):
    """Load a sparse array.

    Path can point towards either a numpy file or a scipy' file saved using
    save_as_sparse().
    """
    indices = np.load(path)
    shape = (index.size for index in indices)
    dense = np.zeros(shape, dtype=int)
    dense[indices] = 1
    return dense
