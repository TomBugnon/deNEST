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


def format_recorder(gid,
                    recorder_type=None,
                    shape=None,
                    locations=None,
                    all_variables=None,
                    all_unit_indices=None):
    """Return the formatted activity of the recorder for all vars and units.

    Return:
        (dict): Dict of the form::
            {`var_1`: [`formatted_activity_unit_index_0`,
                       `formatted_activity_unit_index_1`, ...],
             `var_2`: ...
            }
            where the formatted arrays have the dimensions `shape`.
    """

    if recorder_type == 'multimeter':

        # mm_data = {'var_i': `activity_list`, ...}
        time, sender_gid, mm_data = gather_raw_data(
            gid, all_variables=all_variables, recorder_type='multimeter')
        all_recorder_activity = format_mm_data(
            sender_gid,
            time,
            mm_data,
            locations,
            shape=shape,
            all_variables=all_variables,
            all_unit_indices=all_unit_indices)

    if recorder_type == 'spike_detector':
        time, sender_gid = gather_raw_data(gid, recorder_type='spike_detector')
        all_recorder_activity = format_sd_data(
            sender_gid,
            time,
            locations,
            shape=shape,
            all_unit_indices=all_unit_indices)

    return all_recorder_activity


def format_mm_data(sender_gid,
                   time,
                   mm_data,
                   location_by_gid,
                   shape=None,
                   all_variables=("V_m", ),
                   all_unit_indices=(0, )):
    """Return dict containiing all formatted (t, row, col)-np.arrays."""
    all_recorder_activity = {
        var: [np.zeros(shape) for i in all_unit_indices]
        for var in all_variables
    }
    for (i, t) in enumerate(time):
        row, col, index = location_by_gid[int(sender_gid[i])]
        for var in all_variables:
            all_recorder_activity[var][index][int(t) - 1, row, col] = \
                mm_data[var][i]
    return all_recorder_activity


def format_sd_data(sender_gid,
                   time,
                   location_by_gid,
                   shape=None,
                   all_unit_indices=(0, )):
    """Return dict containing all formatted (t, row, col)-np.arrays."""
    all_recorder_activity = {
        'spikes': [np.zeros(shape) for i in all_unit_indices]
    }
    for (i, t) in enumerate(time):
        row, col, index = location_by_gid[int(sender_gid[i])]
        all_recorder_activity['spikes'][index][int(t) - 1, row, col] = 1.0
    return all_recorder_activity


def gather_raw_data(rec_gid, all_variables=('V_m', ), recorder_type=None):
    """Return non - formatted activity of a given variable saved by the recorder.

    Args:
        rec_gid(tuple): Recorder's NEST GID. Singleton tuple of int.
        variable(str): Variable recorded that we return. Used only for
            multimeters.
        recorder_type(str): 'multimeter' or 'spike_detector'

    Returns:
        tuple: Tuple of the form
            - ( < time > , < sender_gid > , < mm_data > ) for a multimeter, where
                <time> and <sender_gid> are 1D arrays and <mm_data> is a dict
                of 1D arrays containing the activity for each variable in
                `all_variables`.
            - (< time > , < sender_gid > ) for a spike detector.

    """
    record_to = nest.GetStatus(rec_gid, 'record_to')[0]

    if 'memory' in record_to:

        data = nest.GetStatus(rec_gid, 'events')[0]
        time = data['times']
        sender_gid = data['senders']

        if recorder_type == 'multimeter':
            mm_data = {var: data[var] for var in all_variables}
            return (time, sender_gid, mm_data)
        elif recorder_type == 'spike_detector':
            return (time, sender_gid)

    elif 'file' in record_to:
        recorder_files = nest.GetStatus(rec_gid, 'filenames')[0]
        data = load_and_combine(recorder_files)
        time = data[:, 1]
        sender_gid = data[:, 0]

        if recorder_type == 'multimeter':
            all_recorded_variables = nest.GetStatus(rec_gid, 'record_from')[0]
            mm_data = {}
            for var in all_variables:
                variable_col = 2 + all_recorded_variables.index(var)
                mm_data[var] = data[:, variable_col]
            return (time, sender_gid, mm_data)
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
