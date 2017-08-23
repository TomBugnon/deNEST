#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# input_encoding.py

"""Utility functions for checking whether inputs have been encoded."""

import itertools

import numpy as np


def test_network_predictions(test_activity, test_labels, best_label,
                             time_per_movie):

    units_per_label = get_best_label_units(best_label)

    predictions = []
    for movie_activity, label in slice_activity(test_activity, test_labels,
                                                time_per_movie):
        predicted_label = prediction(movie_activity, units_per_label)
        predictions.append(predicted_label == label)
    return sum(predictions)/len(predictions)


def slice_activity(activity, labels, time_per_movie):
    """Slice full activity in list of activity during presentation of a movie.

    Return:
        list: List of tuples each of the form::
                (<movie_activity>, <movie_label>)
            Containing a slice of the full network activity corresponding to
            the presentation of a single movie.

    """
    ntimesteps, _, _ = activity.shape
    # Sanity checks
    assert ntimesteps % time_per_movie == 0

    all_movie_activities = []
    for movie_i in range(int(ntimesteps/time_per_movie)):
        movie_indices = range(movie_i * time_per_movie,
                              (movie_i + 1) * time_per_movie)
        movie_labels = labels[movie_indices]
        movie_activity = activity[movie_indices]
        # Sanity check: only one label per movie
        unique_labels = np.unique(movie_labels)
        assert len(unique_labels) == 1
        all_movie_activities.append((movie_activity, unique_labels[0]))

    return all_movie_activities


def get_best_label_units(best_label):
    """Return dictionary containing units for each preferred label.

    Return:
        dict: Dictionary of the form::
            {'label': <list_of_locs>}

    """
    nrows, ncols = best_label.shape
    units = {label: [] for label in np.unique(best_label)}
    for row, col in itertools.product(range(nrows), range(ncols)):
        units[best_label[row, col]].append((row, col))
    return units

def mean_activity_per_label(activity, labels):
    """Return for each unit the probability of spiking for each label.

    Args:
        activity (np-array): Array of 0 and 1 containing spiking activity for a
            given population. Dimensions (<ntimesteps> * <nrows> * <ncol>).
        labels (seq): Label for each timestep. 1d-array or list of length
            <ntimesteps>

    Return:
        structured np.array: Structured array of dim (<nrows>*<ncols>) of which
            the fields are the unique labels and the values for each location
            and each field contains the probability of firing for a single unit
            during all the timesteps corresponding to that label.

    """
    fields = [(label, float) for label in set(labels)]
    spiking_probs = np.empty((np.size(activity, 1),
                             np.size(activity, 2)),
                             dtype=fields)
    for label in set(labels):
        label_times = [x == label for x in labels]
        spiking_probs[label] = np.mean(activity[label_times, :, :], axis=0)
    return  spiking_probs


def get_best_label(mean_per_label):
    """Return the label for which a unit has most chance of firing.

    Args:
        mean_per_label (np.array): structured array containing the
            probability of firing for each unit and label.

    Return:
        tuple (np-array, np-array): First element is an array of labels,
            second is an array of probability of firing for that label (maximal
            over all possible labels).

    """
    nrows, ncols = mean_per_label.shape
    labels = mean_per_label.dtype.names
    max_labels = np.empty((nrows, ncols), dtype=object)
    for i, j in itertools.product(range(nrows), range(ncols)):
        unit_probs = mean_per_label[i, j].tolist()
        max_labels[i, j] = labels[unit_probs.index(max(unit_probs))]
    return max_labels


def diff_best_to_all(mean_per_label):
    """Return the activity difference between preferred and non-preferred label.

    Args:
        mean_per_label (structured array): output of mean_activity_per_label.
            Structured array containing the probability of firing for each unit
            and label.
        best_label (np-array): Array containing the preferred label for each
            unit.

    Return:
        np-array: Array of the same size as the population containing for each
            unit the difference between the probability of spiking for the
            preferred label, vs all other labels.

    """
    max_label, max_prob = get_best_label(mean_per_label)

    nrows, ncols = mean_per_label.shape
    labels = mean_per_label.dtype.names
    diff_prob = np.empty((nrows, ncols))

    for i, j in itertools.product(range(nrows), range(ncols)):
        unit_probs = mean_per_label[i, j].tolist()
        non_preferred_probs = [p for ind, p in enumerate(unit_probs)
                               if labels[ind] != max_label[i, j]]
        diff_prob[i, j] = max_prob[i, j] - np.mean(non_preferred_probs)

    return diff_prob


def prediction(activity, units_per_label):
    """Return the 'prediction' of the net given activity and preferred labels.

    Args:
        activity (np.array): (t x nrows x ncols) activity array
        units_per_label (dict): dictionary of the form::
            {<label>: <list_of_locs>}
            containing the units for each
            preferred label
    """
    labels = list(units_per_label.keys())

    label_activity = [mean_subpopulation_activity(activity,
                                                  units_per_label[label])
                      for label in labels]

    max_i = label_activity.index(max(label_activity))
    return labels[max_i]


def mean_subpopulation_activity(activity, list_of_locs):
    """Return mean activity of the subpopulation described by locations list."""
    return np.mean([np.mean(activity[:, i, j]) for i, j in list_of_locs])
