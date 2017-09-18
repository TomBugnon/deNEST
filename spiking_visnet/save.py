#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

import os
from os.path import exists, isfile, join

import numpy as np
import yaml

from .utils.sparsify import load_as_numpy


def load_session_times(output_dir):
    """Load session time from output dir."""
    return load_yaml(output_dir, 'session_times')


def load_session_stim(output_dir, session_name):
    """Load full stimulus of a session."""
    movie_prefix = movie_filename(session_name)
    movie_filenames = [f for f in os.listdir(output_dir)
                       if f.startswith(movie_prefix)]
    return load_as_numpy(join(output_dir, movie_filenames[0]))


def load_activity(output_dir, layer, population, variable='spikes',
                  session=None, all_units=False):
    """Load activity of a given variable for a population."""
    if all_units:
        filename_prefix = recorder_filename(layer, population,
                                            variable=variable, unit_index=None)
    else:
        filename_prefix = recorder_filename(layer, population,
                                            variable=variable, unit_index=0)
    print(filename_prefix)
    all_filenames = [f for f in os.listdir(output_dir)
                     if f.startswith(filename_prefix)
                     and isfile(join(output_dir, f))]

    print(all_filenames)
    # Concatenate along first dimension (row)
    all_sessions_activity = np.concatenate(
        [load_as_numpy(join(output_dir, filename))
         for filename in all_filenames],
        axis=1
        )
    print(all_sessions_activity.shape)
    if session is None:
        return  all_sessions_activity
    session_times = load_session_times(output_dir)
    print(session_times)
    return all_sessions_activity[session_times[session]]


def load_labels(output_dir, session_name):
    """Load labels of a session."""
    labels_filename = labels_filename(session_name)
    return np.load(join(output_dir, labels_filename))


def save_as_yaml(path, tree):
    """Save <tree> as yaml file at <path>."""
    with open(path, 'w') as f:
        yaml.dump(tree, f, default_flow_style=False)


def load_yaml(*args):
    """Load yaml file from joined (os.path.join) arguments.

    Return empty list if the file doesn't exist.
    """
    file = join(*args)
    if exists(file):
        with open(join(*args), 'rt') as f:
            return yaml.load(f)
    else:
        return []

def connection_filename(connection):
    """Generate string describing a population-to-population connection."""
    return ('synapses_from_' + connection['source_layer'] + STRING_SEPARATOR
            + connection['source_population'] + '_to_'
            + connection['target_layer'] + STRING_SEPARATOR
            + connection['target_population'])

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
