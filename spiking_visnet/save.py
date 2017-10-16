#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

import os
import pickle
from os.path import exists, isdir, isfile, join

import numpy as np
import scipy.sparse
import yaml

# Use LIL as the default sparse format
sparse_format = scipy.sparse.lil_matrix

# Modify along with FILENAME_FUNCS dict (see end of file)
OUTPUT_SUBDIRS = {'raw': ('raw',), # Raw recorder data (NEST output)
                  'recorders': ('recorders',), # Formatted recorder data
                  'connections': ('connections',), # NEST connection plots
                  'dump': ('dump',), # Network dump
                  'movie': ('sessions',),
                  'labels': ('sessions',),
                  'metadata': ('sessions',),
                  'session_times': ('sessions',),
                  'rasters': ('rasters',),
                  'params': (),
}

# Subdirectories that are cleared if the simulation parameter 'clear_output_dir'
# is set to true.
CLEAR_SUBDIRS = [(), ('recorders',), ('sessions'), ('connections',), ('dump',),
                 ('rasters',)]

def output_subdir(output_dir, data_keyword):
    """Create and return the output subdirectory where a data type is saved.

    Args:
        output_dir (str): path to the main output directory for a simulation.
        data_keyword (str): String designating the type of data for which we
            return an output subdirectory. Should be a key of the OUTPUT_SUBDIRS
            dictionary.
    """
    subdir = join(output_dir, *OUTPUT_SUBDIRS[data_keyword])
    os.makedirs(subdir, exist_ok=True)
    return subdir


def output_filename(data_keyword, *args, **kwargs):
    """Return the filename under which a type of data is saved.

    Args:
        data_keyword (str): String designating the type of data for which we
            return a filename.
        *args: Optional arguments passed to the function generating a filename
            for a given data type.
    """
    return FILENAME_FUNCS[data_keyword](*args, **kwargs)


def output_path(output_dir, data_keyword, *args, **kwargs):
    return join(output_subdir(output_dir, data_keyword),
                output_filename(data_keyword, *args, **kwargs))


def make_output_dir(output_dir, clear_output_dir):
    """Create and possibly clear output directory.

    Create the directory if it doesn't exist and delete the files in the
    subdirectories specified in CLEAR_SUBDIRS if ``clear_output_dir`` is
    True.
    """
    # clear_output_dir = False
    os.makedirs(output_dir, exist_ok=True)
    if clear_output_dir:
        for clear_dir in [join(output_dir, *subdir)
                          for subdir in CLEAR_SUBDIRS
                          if isdir(join(output_dir, *subdir))]:
            print(f'-> Clearing {clear_dir}')
            for f in os.listdir(clear_dir):
                path = join(clear_dir, f)
                if os.path.isfile(path):
                    os.remove(path)


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
    return load_yaml(output_subdir(output_dir, 'sessions'),
                     output_filename('session_times'))


def load_session_stim(output_dir, session_name):
    """Load full stimulus of a session."""
    movie_prefix = output_filename('movie', session_name)
    sessions_dir = output_subdir(output_dir, 'sessions')
    movie_filenames = [f for f in os.listdir(sessions_dir)
                       if f.startswith(movie_prefix)]
    return load_as_numpy(join(sessions_dir, movie_filenames[0]))


def load_activity(output_dir, layer, population, variable='spikes',
                  session=None, all_units=False):
    """Load activity of a given variable for a population."""
    # Get all filenames for that population (one per unit index)
    formatted_dir = output_subdir(output_dir, 'formatted')
    unit_index = None if all_units else 0
    filename_prefix = output_filename('recorders', layer, population,
                                      variable=variable,
                                      unit_index=unit_index)
    all_filenames = [f for f in os.listdir(formatted_dir)
                     if f.startswith(filename_prefix)
                     and isfile(join(formatted_dir, f))]
    # Load the activity for the required unit indices and concatenate along the
    # first dimension.
    # Concatenate along first dimension (row)
    all_sessions_activity = np.concatenate(
        [load_as_numpy(join(formatted_dir, filename))
         for filename in all_filenames],
        axis=1
    )
    # Possibly extract the times for a specific session
    if session is None:
        return  all_sessions_activity
    session_times = load_session_times(output_dir)
    return all_sessions_activity[session_times[session]]


def load_labels(output_dir, session_name):
    """Load labels of a session."""
    return np.load(join(output_subdir(output_dir, 'sessions'),
                        output_filename('labels', 'session_name')))


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


def session_times_filename():
    return 'session_times'


def params_filename():
    return 'params'


def rasters_filename(layer, pop):
    return 'spikes_raster_' + layer + '_' + pop + '.png'


FILENAME_FUNCS = {'params': params_filename,
                  'session_times': session_times_filename,
                  'metadata': metadata_filename,
                  'labels': labels_filename,
                  'movie': movie_filename,
                  'recorders': recorder_filename,
                  'rasters': rasters_filename
}
