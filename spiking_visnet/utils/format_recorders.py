#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# format_recorders.py


import numpy as np


def format_mm_data(sender_gid, time, activity, location_by_gid, dim=None):
    """Return (row, col, t)-np.array from non-formatted multimeter data."""
    activity_array = np.zeros(dim)
    for (i, t) in enumerate(time):

        row, col = location_by_gid[int(sender_gid[i])]
        activity_array[row, col, int(t)] = activity[i]

    return activity_array


def format_sd_data(sender_gid, time, location_by_gid, dim=None):
    """Return (row, col, t)-np.array from non-formatted spike_detector data."""
    activity_array = np.zeros(dim)

    for (i, t) in enumerate(time):
        row, col = location_by_gid[int(sender_gid[i])]
        activity_array[row, col, int(t)] = 1

    return activity_array
