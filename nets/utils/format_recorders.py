#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# format_recorders.py
"""Format NEST recorders activity in (time*row*col) numpy arrays.

NB: If multiple recorded units (of a given population) are at the same location
in the layer, we save only one of those.
"""

from os import stat
from os.path import isfile

import nest
import numpy as np


def format_recorder(gid, recorder_type=None, shape=None, locations=None,
                    variable=None, unit_index=None):

    if recorder_type == 'multimeter':

        time, sender_gid, activity = gather_raw_data(gid, variable,
                                                     recorder_type='multimeter')
        activity_array = format_mm_data(sender_gid, time, activity, locations,
                                        shape=shape, unit_index=unit_index)

    if recorder_type == 'spike_detector':
        time, sender_gid = gather_raw_data(gid, recorder_type='spike_detector')
        activity_array = format_sd_data(sender_gid, time, locations,
                                        shape=shape, unit_index=unit_index)

    return activity_array


def format_mm_data(sender_gid, time, activity, location_by_gid, shape=None,
                   unit_index=0):
    """Return (t, row, col)-np.array from non-formatted multimeter data."""
    activity_array = np.zeros(shape)
    for (i, t) in enumerate(time):
        row, col, index = location_by_gid[int(sender_gid[i])]
        if index == unit_index:
            activity_array[int(t) - 1, row, col] = activity[i]

    return activity_array


def format_sd_data(sender_gid, time, location_by_gid, shape=None, unit_index=0):
    """Return (t, row, col)-np.array from non-formatted spike_detector data."""
    activity_array = np.zeros(shape)

    for (i, t) in enumerate(time):
        row, col, index = location_by_gid[int(sender_gid[i])]
        if index == unit_index:
            activity_array[int(t) - 1, row, col] = 1

    return activity_array


def gather_raw_data(rec_gid, variable='V_m', recorder_type=None):
    """Return non - formatted activity of a given variable saved by the recorder.

    Args:
        rec_gid(tuple): Recorder's NEST GID. Singleton tuple of int.
        variable(str): Variable recorded that we return. Used only for
            multimeters.
        recorder_type(str): 'multimeter' or 'spike_detector'

    Returns:
        tuple: Tuple of 1d np.arrays of the form
            - ( < time > , < sender_gid > , < activity > ) for a multimeter, where
                activity is that of the variable < variable > .
            - (< time > , < sender_gid > ) for a spike detector.

    """
    record_to = nest.GetStatus(rec_gid, 'record_to')[0]

    if 'memory' in record_to:

        data = nest.GetStatus(rec_gid, 'events')[0]
        time = data['times']
        sender_gid = data['senders']

        if recorder_type == 'multimeter':
            activity = data[variable]
            return (time, sender_gid, activity)
        elif recorder_type == 'spike_detector':
            return (time, sender_gid)

    elif 'file' in record_to:
        recorder_files = nest.GetStatus(rec_gid, 'filenames')[0]
        data = load_and_combine(recorder_files)
        time = data[:, 1]
        sender_gid = data[:, 0]

        if recorder_type == 'multimeter':
            # Get proper column
            all_variables = nest.GetStatus(rec_gid, 'record_from')[0]
            variable_col = 2 + all_variables.index(variable)
            activity = data[:, variable_col]
            return (time, sender_gid, activity)
        elif recorder_type == 'spike_detector':
            return (time, sender_gid)


def load_and_combine(recorder_files_list):
    """Load the recorder data from files.

    Args:
        recorder_files_list(list): List of absolute paths to the files in
            which NEST saved a single recorder's activity.

    Returns:
        (np.array): Array of which columns are the files' columns and rows are
            the events recorded in the union of all files. If all the files are
            empty or there is no filename, returns an array with 0 rows.
            Array np - loaded in each text file is enforced to have two
            dimensions. If no data is found at all, returns an array with zero
            rows.

    """
    file_data_list = [
        np.loadtxt(filepath, dtype=float, ndmin=2)
        for filepath in recorder_files_list
        if isfile(filepath) and not stat(filepath).st_size == 0
    ]

    if file_data_list:
        return np.concatenate(file_data_list, axis=0)
    return np.empty((0, 10))
