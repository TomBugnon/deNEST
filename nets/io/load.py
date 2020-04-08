#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# load.py

"""Utility functions for data loading."""

import os
from os.path import dirname, exists, join

import pandas as pd
import yaml


def load_yaml(*args):
    """Load yaml file from joined (os.path.join) arguments.

    Return empty list if the file doesn't exist.
    """
    path = join(*args)  # pylint:disable=no-value-for-parameter
    if exists(path):
        with open(join(*args), 'rt') as f:  # pylint:disable=no-value-for-parameter
            return yaml.load(f, Loader=yaml.SafeLoader)
    else:
        return []


def load(metadata_path):
    """Load tabular data from metadata file and return a pandas df.

    The data files are assumed to be in the same directory as the metadata.

    Args:
        metadata_path (str or Path): Path to the yaml file containing the
            metadata for a recorder.

    Returns:
        pd.DataFrame : pd dataframe containing the raw data, possibly
            subsampled. Columns may be dropped ( see `usecols` kwarg) and 'x',
            'y' 'z' location fields may be added (see `assign_locations` kwarg).
    """
    print(f"Loading {metadata_path}")
    metadata = load_yaml(metadata_path)
    filepaths = get_filepaths(metadata_path)

    return load_as_df(metadata['colnames'], *filepaths)


def load_as_df(colnames, *paths, sep='\t', **kwargs):
    """Load tabular data from one or more files and return a pandas df.

    Keyword arguments are passed to ``pandas.read_csv()``.

    Arguments:
        colnames (tuple[str]): The names of the columns.
        *paths (filepath or buffer): The file(s) to load data from.

    Kwargs:
        **kwargs: Passed to pd.read_csv

    Returns:
        pd.DataFrame: The loaded data.
    """

    # Read data from disk
    return pd.concat(
        [
            pd.read_csv(path, sep=sep, **kwargs)
            for path in paths
        ]
    )


def load_metadata_filenames(output_dir, recorder_type):
    """Return list of all recorder filenames for a certain recorder type"""
    if recorder_type not in ['multimeter', 'spike_detector', 'weight_recorder']:
        raise ValueError("Recorder type not recognized")
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
    ])  # TODO


def get_filepaths(metadata_path):
    metadata = load_yaml(metadata_path)
    # Check loaded metadata
    assert 'filenames' in metadata
    # We assume metadata and data are in the same directory
    return [join(dirname(metadata_path), filename)
            for filename in metadata['filenames']]


def load_session_times(output_dir):
    """Load session time from output dir."""
    return load_yaml(output_path(output_dir, 'session_times'))
