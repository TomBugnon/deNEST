#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# input_encoding.py

"""Utility functions to get statistics about units' activity"""


import itertools

import numpy as np


def firing_rate(activity):
    """Return an array containing the firing rate in Hz."""
    return 1000 * np.sum(activity, axis=0)/np.size(activity, axis=0)


def isi(spike_train):
    """Return the inter-spike intervals of a list-like of 0 and 1."""
    spike_times = [t for t, spike in enumerate(spike_train) if spike]
    return [spike_times[i + 1] - spike_times[i]
            for i in range(len(spike_times) - 1)]

def all_isi(activity):
    """Return an array of lists of inter-spike intervals from activity array."""
    _, nrows, ncols = activity.shape
    isis = np.empty((nrows, ncols), dtype=list)
    for row, col in itertools.product(range(nrows), range(ncols)):
        isis[row, col] = isi(activity[:, row, col])
    return isis

def all_cv(activity):
    """Return array of coefficients of variation."""
    _, nrows, ncols = activity.shape
    cvs = np.zeros((nrows, ncols))
    for row, col in itertools.product(range(nrows), range(ncols)):
        inter_spike_list = isi(activity[:, row, col])
        cvs[row, col] = float(np.std(inter_spike_list))/float(np.mean(inter_spike_list))
    return cvs
