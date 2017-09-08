#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sparsify.py

"""Save and load binary arrays sparsely."""

import os
import os.path
import pickle

import numpy as np
import scipy.sparse

# Use LIL as the default sparse format
sparse_format = scipy.sparse.lil_matrix


def ensure_ext(path, ext='.pkl'):
    """Add a file extension if there isn't one."""
    path, _ext = os.path.splitext(path)
    _ext = _ext or ext
    return path + _ext


# TODO: rename to save_sparse
def save_as_sparse(path, array):
    """Save an array in a sparse format."""
    # Normalize file extension
    path = ensure_ext(path, ext='.pkl')
    # Store shape
    shape = array.shape
    # Ensure 2D
    array = array.reshape(shape[0], -1)
    # Save
    data = {'shape': shape,
            'data': sparse_format(array)}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return True


def load_as_numpy(path):
    """Load as numpy a file saved with ``save_as_sparse`` or ``np.save``."""
    ext = os.path.splitext(path)[1]
    if ext == '.npy':
        return np.load(path)
    return load_sparse(path)


def load_sparse(path):
    """Load an array saved with ``save_as_sparse``."""
    # Normalize file extension
    path = ensure_ext(path, ext='.pkl')
    # Load
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    shape, data = loaded['shape'], loaded['data']
    # Convert to dense
    data = data.toarray()
    # Reshape
    return data.reshape(shape)
