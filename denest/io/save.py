#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py

"""Utility functions for data saving."""

import logging
import shutil
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# Modify along with FILENAME_FUNCS dict (see end of file)
OUTPUT_SUBDIRS = {
    "tree": (),
    "versions": (),
    "raw_data": ("data",),  # Raw recorder data (NEST output)
    # Metadata for recorders (contains filenames and gid/location mappings)
    "recorders_metadata": ("data",),
    "projection_recorders_metadata": ("data",),
    "session_times": (),
}

# Subdirectories that are cleared during OUTPUT_DIR initialization
CLEAR_SUBDIRS = [subdir for subdir in OUTPUT_SUBDIRS.values()]


def save_as_yaml(path, tree):
    """Save <tree> as yaml file at <path>."""
    path = Path(path).with_suffix(".yml")
    with open(path, "w") as f:
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

    Keyword Args:
        create_dir (bool): If true, the returned directory is created.
    """
    path = Path(output_dir, *OUTPUT_SUBDIRS[data_keyword])
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    return path


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
    return Path(
        output_subdir(output_dir, data_keyword),
        output_filename(data_keyword, *args, **kwargs),
    )


def make_output_dir(output_dir, clear_output_dir=True, delete_subdirs_list=None):
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
    output_dir = Path(output_dir)
    if delete_subdirs_list is None:
        delete_subdirs_list = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_output_dir:
        for path in [Path(output_dir, *subdir) for subdir in CLEAR_SUBDIRS]:
            if path.exists():
                log.info("Clearing directory: %s", path)
                # Delete files in the CLEAR_SUBDIRS
                delete_files(path)
                # Delete the contents of all the delete_subdirs we encounter
                delete_subdirs(path, delete_subdirs_list)


def _delete_contents(path, to_delete="files", missing_ok=True):
    if missing_ok and not path.is_dir():
        return path
    for child in path.iterdir():
        if to_delete == "files":
            if child.is_file():
                child.unlink()
        elif child.is_dir() and child in to_delete:
            shutil.rmtree(path)
    return path


def delete_files(path, missing_ok=True):
    """Delete all files in a directory.

    Not recursive. Only deletes files, not subdirectories.
    """
    return _delete_contents(path, to_delete="files", missing_ok=missing_ok)


def delete_subdirs(path, to_delete, missing_ok=True):
    """Delete some subdirectories in a directory.

    Not recursive. Only deletes the subdirectories in ``to_delete``.
    """
    return _delete_contents(path, to_delete=to_delete, missing_ok=missing_ok)


def recorder_metadata_filename(label):
    """Return filename for a recorder from its label."""
    return label


def metadata_filename():
    return "session_metadata.yml"


def session_times_filename():
    return "session_times.yml"


def tree_filename():
    return "parameter_tree.yml"


def rasters_filename(layer, pop):
    return "spikes_raster_" + layer + "_" + pop + ".png"


def version_info_filename():
    return "versions.txt"


FILENAME_FUNCS = {
    "tree": tree_filename,
    "recorders_metadata": recorder_metadata_filename,
    "session_times": session_times_filename,
    "session_metadata": metadata_filename,
    "versions": version_info_filename,
}
