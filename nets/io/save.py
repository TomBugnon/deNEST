#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Utility functions for data saving."""

# pylint: disable=missing-docstring,ungrouped-imports

import os
import shutil
from os.path import isdir, join
from pathlib import Path

import yaml

# Modify along with FILENAME_FUNCS dict (see end of file)
OUTPUT_SUBDIRS = {
    'params': (),
    'git_hash': (),
    'raw_data': ('data',),  # Raw recorder data (NEST output)
    # Metadata for recorders (contains filenames and gid/location mappings)
    'recorders_metadata': ('data',),
    'connection_recorders_metadata': ('data',),
    'session_times': (),
    'dump': ('network_dump',),  # Network dump
    'rasters': ('rasters',),
    'plots': ('plots',),
}

# Subdirectories that are cleared during OUTPUT_DIR initialization
CLEAR_SUBDIRS = [subdir for subdir in OUTPUT_SUBDIRS.values()]


def save_as_yaml(path, tree):
    """Save <tree> as yaml file at <path>."""
    path = Path(path).with_suffix('.yml')
    with open(path, 'w') as f:
        yaml.dump(tree, f, default_flow_style=False)

#
# Paths, filenames and output directory organisation
#


def output_subdir(output_dir, data_keyword, create_dir=True):
    """Create and return the output subdirectory where a data type is saved.

    Args:
        output_dir (str): path to the main output directory for a simulation.
        data_keyword (str): String designating the type of data for which we
            return an output subdirectory. Should be a key of the OUTPUT_SUBDIRS
            dictionary.
    
    Kwargs:
        create_dir (bool): If true, the returned directory is created.
    """
    subdir = join(output_dir, *OUTPUT_SUBDIRS[data_keyword])
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


def output_path(output_dir, data_keyword, *args, **kwargs):
    """Return the full path at which an object is saved."""
    return join(output_subdir(output_dir,
                              data_keyword),
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
            print(f'-> Clearing directory {clear_dir}')
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
