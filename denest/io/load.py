#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# load.py

"""Utility functions for data loading."""

import logging
from pathlib import Path

import pandas as pd
import yaml

from .save import output_path, output_subdir

log = logging.getLogger(__name__)


def load_session_times(output_dir):
    """Load session time from output dir."""
    return load_yaml(output_path(output_dir, "session_times"))


def metadata_paths(output_dir):
    """Return list of paths to all recorder metadata files."""
    metadata_dir = output_subdir(output_dir, "recorders_metadata", create_dir=False)
    # "metadata" files are the ones without .ext
    return sorted(metadata_dir.glob("*.yml"))


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
    log.info("Loading metadata from %s", metadata_path)
    metadata = load_yaml(metadata_path)
    filepaths = get_filepaths(metadata_path)

    return load_as_df(metadata["colnames"], *filepaths)


def load_as_df(colnames, *paths, sep="\t", index_col=False, header=None, **kwargs):
    """Load tabular data from one or more files and return a pandas df.

    Keyword arguments are passed to ``pandas.read_csv()``.

    Arguments:
        colnames (tuple[str]): The names of the columns.
        *paths (filepath or buffer): The file(s) to load data from.

    Keyword Args:
        **Keyword Args: Passed to pd.read_csv

    Returns:
        pd.DataFrame: The loaded data.
    """

    if not paths:
        return pd.DataFrame()
    # Read data from disk
    return pd.concat(
        [
            pd.read_csv(
                path,
                names=colnames,
                sep=sep,
                index_col=index_col,
                header=header,
                **kwargs
            )
            for path in paths
        ]
    )


def get_filepaths(metadata_path):
    metadata_path = Path(metadata_path)
    metadata = load_yaml(metadata_path)
    # Check loaded metadata
    assert "filenames" in metadata
    # We assume metadata and data are in the same directory
    return [metadata_path.parent / filename for filename in metadata["filenames"]]


def load_yaml(*args):
    """Load a YAML file from a path."""
    path = Path(*args)
    with path.open("rt") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
