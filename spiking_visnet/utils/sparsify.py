#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sparsify.py


"""Compress, save and load 3D and 4D np arrays."""


import numpy as np
from scipy import sparse


# TODO:
def save_as_sparse(path, nparray):
    """Save 3D or 4D nparrays as scipy sparse array in <path>."""
    np.save(path, nparray)


# TODO:
def load_as_numpy(path):
    """Load file at <path> as np array.

    Path can point towards either a numpy file or a 'scipy' file saved using
    save_as_sparse().
    """
    return np.load(path)
