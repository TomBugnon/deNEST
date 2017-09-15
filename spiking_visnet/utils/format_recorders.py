#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# format_recorders.py

"""Format NEST recorders activity in (time*row*col) numpy arrays.

NB: If multiple recorded units (of a given population) are at the same location
in the layer, we save only one of those.
"""

import numpy as np


def format_mm_data(sender_gid, time, activity, location_by_gid, dim=None,
                   unit_index=0):
    """Return (t, row, col)-np.array from non-formatted multimeter data."""
    activity_array = np.zeros(dim)
    for (i, t) in enumerate(time):
        row, col, index = location_by_gid[int(sender_gid[i])]
        if index == unit_index:
            activity_array[int(t), row, col] = activity[i]

    return activity_array


def format_sd_data(sender_gid, time, location_by_gid, dim=None, unit_index=0):
    """Return (t, row, col)-np.array from non-formatted spike_detector data."""
    activity_array = np.zeros(dim)

    for (i, t) in enumerate(time):
        row, col, index = location_by_gid[int(sender_gid[i])]
        if index == unit_index:
            activity_array[int(t), row, col] = 1

    return activity_array
