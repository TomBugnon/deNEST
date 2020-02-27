#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

# pylint: disable=missing-docstring,ungrouped-imports

import os
import pickle
import shutil
from os.path import dirname, exists, isdir, join

import numpy as np
import pandas as pd
import scipy.sparse
import yaml

# Use LIL as the default sparse format
sparse_format = scipy.sparse.lil_matrix  # pylint:disable=invalid-name

# Modify along with FILENAME_FUNCS dict (see end of file)
OUTPUT_SUBDIRS = {
    'params': (),
    'git_hash': (),
    'raw_data': ('data',),  # Raw recorder data (NEST output)
    'recorders_metadata': ('data',),  # Metadata for recorders (contains filenames and gid/location mappings)
    'connection_recorders_metadata': ('data',),
    'session_movie': ('sessions',),
    'session_labels': ('sessions',),
    'session_metadata': ('sessions',),
    'session_times': ('sessions',),
    'connections': ('connection_plots',),  # NEST connection plots
    'dump': ('network_dump',),  # Network dump
    'rasters': ('rasters',),
    'plots': ('plots',),
    'measures': ('measures',),
}

# Subdirectories that are cleared if the simulation parameter 'clear_output_dir'
# is set to true.
CLEAR_SUBDIRS = [subdir for subdir in OUTPUT_SUBDIRS.values()]


################################
#### Utility functions for data loading
################################


def load_dict(path):
    """Load a big dic with pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_as_numpy(path):
    """Load as numpy a file saved with ``save_array`` or ``np.save``."""
    ext = os.path.splitext(path)[1]
    if ext == '.npy':
        return np.load(path)
    return load_sparse(path)


def load_yaml(*args):
    """Load yaml file from joined (os.path.join) arguments.

    Return empty list if the file doesn't exist.
    """
    path = join(*args)  # pylint:disable=no-value-for-parameter
    if exists(path):
        with open(join(*args), 'rt') as f:  # pylint:disable=no-value-for-parameter
            return yaml.load(f)
    else:
        return []


def load(metadata_path, assign_locations=False, usecols=None, filter_ratio=None,
         filter_type=None):
    """Load tabular data from metadata file and return a pandas df.

    The data files are assumed to be in the same directory as the metadata.

    Args:
        metadata_path (str or Path): Path to the yaml file containing the
            metadata for a recorder.

    Kwargs:
        all_unit_indices (bool): If false, we index the returned array by z=0
            where z is the unit index at a given grid location
        assign_locations (bool): If True, add x y and z values indicating gid
            grid position and unit index amongst each population
        usecols (tuple[str] or None): Columns of the data that are loaded. By
            default, all columns are loaded. Can be useful for multimeters if we
            are only interested in one amongst many variables.
        filter_ratio (dict): Dictionary of the form:
                `{<column>: <ratio_of_loaded_values>}`
            used to filter loaded data. For all filtered fields, the unique
            values are loaded and sampled evenly or randomly based on the
            corresponding value of the `filter_type` parameter.
        filter_type (dict): Dictionary of the form:
                `{<column>: <type_of_filtering>}`
            used to filter loaded data. The types of filtering are 'even' or
            'random' (default)

    Returns:
        pd.DataFrame : pd dataframe containing the raw data, possibly
            subsampled. Columns may be dropped ( see `usecols` kwarg) and 'x',
            'y' 'z' location fields may be added (see `assign_locations` kwarg).
    """
    print(f"Loading {metadata_path}, filter_ratio={filter_ratio}, "
          f"filter_type={filter_type}, usecols={usecols}, assign_locations={assign_locations}")
    metadata = load_yaml(metadata_path)
    filepaths = get_filepaths(metadata_path)

    # Loaded columns
    cols = metadata['colnames']
    if usecols is None:
        usecols = cols
    else:
        usecols = list(set(cols) & set(usecols))
    if not usecols:
        return None

    # Target values for each filtered column
    def get_target_values(metadata, *filepaths,
                          usecols=None,
                          filter_ratio=None, filter_type=None):
        if filter_ratio is None:
            filter_ratio = {}
        if filter_type is None:
            filter_type = {}
        filter_values = {}
        for column, proportion in filter_ratio.items():
            unique_values = load_as_df(
                metadata['colnames'],
                *filepaths,
                usecols=[column]
            )[column].unique()
            filt_type = filter_type.get(column, 'random')
            assert filt_type in ['random', 'even']
            if filt_type == 'even':
                # Every 1/proportion values in sorted list of unique values
                filter_values[column] = sorted(
                    unique_values
                )[::int(1/proportion)]
            else:
                filter_values[column] = np.random.choice(
                    unique_values,
                    int(proportion*len(unique_values))
                )
        return filter_values

    filter_values = get_target_values(metadata, *filepaths, usecols=usecols,
                                      filter_ratio=filter_ratio,
                                      filter_type=filter_type)

    data = load_as_df(metadata['colnames'],
                      *filepaths,
                      usecols=usecols,
                      filter_values=filter_values)
    if assign_locations:
        # print('Assigning locations')
        data = assign_gid_locations(data, metadata['locations'])
        # print('Done assigning locations')

    return data


def load_as_df(names, *paths, sep='\t', index_col=False, usecols=None,
               filter_values=None, **kwargs):
    """Load tabular data from one or more files and return a pandas df.

    Keyword arguments are passed to ``pandas.read_csv()``.

    Arguments:
        colnames (tuple[str]): The names of the columns.
        locations (dict of int:tuple): Dictionary containing the gid to
            (x,y,z)-positions mappings for the population. Locations are
            3-tuples of int. First two dimensions correspond to x and y
            location in the layer, 3rd dimension corresponds to the "unit
            index" (if there is more than one unit per grid location)
        *paths (filepath or buffer): The file(s) to load data from.

    Kwargs:
        filter_values (dict): Dictionary of the form:
                `{<column>: <target_values_list>}`
            used to filter loaded data.

    Returns:
        pd.DataFrame: The loaded data.
    """

    # Read data from disk
    return pd.concat(
        filter_df(
            pd.read_csv(path, sep=sep, names=names, index_col=index_col,
                        usecols=usecols, **kwargs),
            filter_values
        )
        for path in paths
    )


def filter_df(df, filter_values):
    """Filter df by matching targets for multiple columns.

    Args:
        df (pd.DataFrame): dataframe
        filter_values (None or dict): Dictionary of the form:
                `{<field>: <target_values_list>}`
            used to filter columns data.
    """
    if filter_values is None or not filter_values:
        return df
    return df[
        np.logical_and.reduce([
            df[column].isin(target_values)
            for column, target_values in filter_values.items()
        ])
    ]


# TODO: This step is slow ! What can we do?
def assign_gid_locations(df, locations):
    """Assign x and y columns (loc in grid), and z (index at grid location)."""
    return df.assign(
        x=df.gid.map(lambda gid: locations[gid][0]),
        y=df.gid.map(lambda gid: locations[gid][1]),
        z=df.gid.map(lambda gid: locations[gid][2])
    )


def load_metadata_filenames(output_dir, recorder_type):
    """Return list of all recorder filenames for a certain recorder type"""
    assert recorder_type in ['multimeter', 'spike_detector', 'weight_recorder']
    if recorder_type in ['multimeter', 'spike_detector']:
        metadata_dir = output_subdir(output_dir,
                                     'recorders_metadata',
                                     create_dir=False)
    elif recorder_type in ['weight_recorder']:
        metadata_dir = output_subdir(output_dir,
                                     'connection_recorders_metadata',
                                     create_dir=False)
    return sorted([
        f for f in os.listdir(metadata_dir)
        if recorder_type in os.path.splitext(f)[0]
        and not os.path.splitext(f)[1]  # "metadata" files are the ones without .ext
    ])


def get_filepaths(metadata_path):
    metadata = load_yaml(metadata_path)
    # Check loaded metadata
    assert 'filenames' in metadata, \
        (f'The metadata loaded at path: `{metadata_path}` does not have the '
         f'correct format. Please check package versions?')
    # We assume metadata and data are in the same directory
    return [join(dirname(metadata_path), filename)
            for filename in metadata['filenames']]


def load_session_times(output_dir):
    """Load session time from output dir."""
    return load_yaml(output_path(output_dir, 'session_times'))


def load_weight_recorder(output_dir, conn_name, start_trim=None):
    path = output_path(output_dir,
                       'connection_recorders',
                       conn_name)
    w_dict = load_dict(path)
    if start_trim is None:
        return w_dict
    time_i = [i for i, t in enumerate(w_dict['times']) if t >= start_trim]
    return {
        key: data[time_i]
        for key, data in w_dict.items()
    }


def load_labels(output_dir, session_name):
    """Load labels of a session."""
    return np.load(join(output_subdir(output_dir, 'sessions'),
                        output_filename('session_labels', session_name)))


def load_measure(output_dir, measure_name, session=None, start_trim=None,
                 end_trim=None):
    """Load previously saved measure (eg LFP)"""
    measure_dir = join(output_subdir(output_dir, 'measures'))
    filenames = [f for f in os.listdir(measure_dir)
                 if f.startswith(measure_name)]
    assert len(filenames) == 1
    measure = load_as_numpy(join(measure_dir, filenames[0]))
    # Possibly extract the times corresponding to a specific session
    if session is not None:
        times = load_session_times(output_dir)[session]
    else:
        times = range(len(measure))
    # Possibly trim the beginning and the end, after selecting for session
    if start_trim is not None:
        times = range(min(times) + int(start_trim), max(times))
    if end_trim is not None:
        times = range(min(times), max(times) - int(end_trim))
    return measure[times]


# TODO: fix pickle.dump failure for large files
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
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except OSError as e:
        msg = (f"Could not save data at {f}! \n"
               f"... Ignoring the following error: {str(e)}`")
        import warnings
        warnings.warn(msg)


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


################################
#### Utility functions for data saving
################################


def save_as_yaml(path, tree):
    """Save <tree> as yaml file at <path>."""
    with open(path, 'w') as f:
        yaml.dump(tree, f, default_flow_style=False)


def save_array(path, array):
    """Save array either as dense or sparse depending on data type."""
    try:
        save_sparse(path, array)
    except TypeError:
        np.save(path, array)


# TODO: fix pickle.dump failure for large files
def save_dict(path, dictionary):
    """Save a big dic with pickle."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f)
    except OSError as e:
        msg = (f"Could not save data at {f}! \n"
               f"... Ignoring the following error: {str(e)}`")
        import warnings
        warnings.warn(msg)


def save_measure(output_dir, measure, measure_name):
    """Save measure as np-array (eg LFP)."""
    path = join(output_subdir(output_dir, 'measures'), measure_name)
    np.save(path, measure)
    return path


def save_plot(fig, output_dir, filename, overwrite=False):
    """Save matplotlib figure in 'plot' subdirectory."""
    filename = filename.replace('.', ',')
    path = join(output_subdir(output_dir, 'plots'), filename)
    if os.path.exists(path) and not overwrite:
        print(f'Not overwriting file at {path}')
        return
    fig.savefig(path)


################################
#### Paths, filenames and output directory organisation
################################


def output_subdir(output_dir, data_keyword, session_name=None, create_dir=True):
    """Create and return the output subdirectory where a data type is saved.

    Args:
        output_dir (str): path to the main output directory for a simulation.
        data_keyword (str): String designating the type of data for which we
            return an output subdirectory. Should be a key of the OUTPUT_SUBDIRS
            dictionary.
        session_name (str or None): If a session is provided, data is organized
            by subdirectories with that session's name.
    """
    if session_name is None:
        subdir = join(output_dir, *OUTPUT_SUBDIRS[data_keyword])
    else:
        subdir = join(output_dir, *OUTPUT_SUBDIRS[data_keyword], session_name)
    if create_dir:
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


def output_path(output_dir, data_keyword, *args, session_name=None, **kwargs):
    return join(output_subdir(output_dir,
                              data_keyword,
                              session_name=session_name),
                output_filename(data_keyword, *args, **kwargs))


def make_output_dir(output_dir, clear_output_dir=True,
                    delete_subdirs_list=None):
    """Create and possibly clear output directory.

    Create the directory if it doesn't exist.
    If `clear_output_dir` is True, we clear the directory. We iterate over all
    the subdirectories specified in CLEAR_SUBDIRS, and for each of those we:
        - delete all the files
        - delete all the directories whose name is in the `delete_subdirs` list.

    Args:
        output_dir (str):
        clear_output_dir (bool): Whether we clear the CLEAR_SUBDIRS
        delete_subdirs_list (list of str or None): List of subdirectories of
            CLEAR_SUBDIRS that we delete.
    """
    if delete_subdirs_list is None:
        delete_subdirs_list = []
    os.makedirs(output_dir, exist_ok=True)
    if clear_output_dir:
        for clear_dir in [join(output_dir, *subdir)
                          for subdir in CLEAR_SUBDIRS
                          if isdir(join(output_dir, *subdir))]:
            print(f'-> Clearing {clear_dir}')
            # Delete files in the CLEAR_SUBDIRS
            delete_files(clear_dir)
            # Delete the contents of all the delete_subdirs we encounter
            delete_subdirs(clear_dir, delete_subdirs_list)


def delete_files(clear_dir):
    """Delete all files in a directory."""
    for f in os.listdir(clear_dir):
        path = join(clear_dir, f)
        if os.path.isfile(path):
            os.remove(path)


def delete_subdirs(clear_dir, delete_subdirs_list):
    """Delete some subdirectories in a directory."""
    for f in os.listdir(clear_dir):
        path = join(clear_dir, f)
        if os.path.isdir(path) and f in delete_subdirs_list:
            shutil.rmtree(path)


def recorder_metadata_filename(label):
    """Return filename for a recorder from its label."""
    return label


def movie_filename():
    return 'session_movie'


def labels_filename():
    return 'session_labels'


def metadata_filename():
    return 'session_metadata'


def session_times_filename():
    return 'session_times'


def params_filename():
    return 'params'


def rasters_filename(layer, pop):
    return 'spikes_raster_' + layer + '_' + pop + '.png'


def git_hash_filename():
    return 'git_hash'


FILENAME_FUNCS = {
    'params': params_filename,
    'recorders_metadata': recorder_metadata_filename,
    'connection_recorders': recorder_metadata_filename,
    'connection_recorders_metadata': recorder_metadata_filename,
    'session_times': session_times_filename,
    'session_metadata': metadata_filename,
    'session_labels': labels_filename,
    'session_movie': movie_filename,
    'rasters': rasters_filename,
    'git_hash': git_hash_filename,
}
