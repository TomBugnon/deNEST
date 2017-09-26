#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

import os
import pickle
from os.path import exists, isfile, join

import numpy as np
import scipy.sparse
import yaml

# Use LIL as the default sparse format
sparse_format = scipy.sparse.lil_matrix


def save_array(path, array):
    """Save array either as dense or sparse depending on data type."""
    try:
        save_sparse(path, array)
    except TypeError:
        np.save(path, array)


def load_as_numpy(path):
    """Load as numpy a file saved with ``save_array`` or ``np.save``."""
    ext = os.path.splitext(path)[1]
    if ext == '.npy':
        return np.load(path)
    return load_sparse(path)


def save_as_yaml(path, tree):
    """Save <tree> as yaml file at <path>."""
    with open(path, 'w') as f:
        yaml.dump(tree, f, default_flow_style=False)


def load_yaml(*args):
    """Load yaml file from joined (os.path.join) arguments.

    Return empty list if the file doesn't exist.
    """
    path = join(*args)
    if exists(path):
        with open(join(*args), 'rt') as f:
            return yaml.load(f)
    else:
        return []


def load_session_times(output_dir):
    """Load session time from output dir."""
    return load_yaml(output_dir, 'session_times')


def load_session_stim(output_dir, session_name):
    """Load full stimulus of a session."""
    movie_prefix = movie_filename(session_name)
    movie_filenames = [f for f in os.listdir(join(output_dir, 'sessions'))
                       if f.startswith(movie_prefix)]
    return load_as_numpy(join(output_dir, 'sessions', movie_filenames[0]))


def load_activity(output_dir, layer, population, variable='spikes',
                  session=None, all_units=False):
    """Load activity of a given variable for a population."""
    if all_units:
        filename_prefix = recorder_filename(layer, population,
                                            variable=variable, unit_index=None)
    else:
        filename_prefix = recorder_filename(layer, population,
                                            variable=variable, unit_index=0)
    all_filenames = [f for f in os.listdir(output_dir)
                     if f.startswith(filename_prefix)
                     and isfile(join(output_dir, f))]

    # Concatenate along first dimension (row)
    all_sessions_activity = np.concatenate(
        [load_as_numpy(join(output_dir, filename))
         for filename in all_filenames],
        axis=1
        )
    if session is None:
        return  all_sessions_activity
    session_times = load_session_times(output_dir)
    return all_sessions_activity[session_times[session]]


def load_labels(output_dir, session_name):
    """Load labels of a session."""
    return np.load(join(output_dir, labels_filename(session_name)))


def save_sparse(path, array):
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


def ensure_ext(path, ext='.pkl'):
    """Add a file extension if there isn't one."""
    path, _ext = os.path.splitext(path)
    _ext = _ext or ext
    return path + _ext


def load_sparse(path):
    """Load an array saved with ``save_array``."""
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


# TODO
def connection_filename(connection):
    """Generate string describing a population-to-population connection."""
    pass


def recorder_filename(layer, pop, unit_index=None, variable='spikes'):
    """Return filename for a population x unit_index."""
    base_filename = (layer + '_' + pop + '_'
                     + variable)
    suffix = ''
    if unit_index is not None:
        suffix = ('_' + 'units' + '_'
                  + str(unit_index))
    return base_filename + suffix

def movie_filename(session_name):
    return 'session_' + session_name + '_movie'

def labels_filename(session_name):
    return 'session_' + session_name + '_labels'

def metadata_filename(session_name):
    return 'session_' + session_name + '_metadata'
