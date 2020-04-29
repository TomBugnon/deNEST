#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/spike_times.py
"""Draw times of spiking from instantaneous rates."""

import itertools

import numpy as np


def draw_spike_times(all_instantaneous_rates, start_time=0, distribution="poisson"):
    """Draw spike times from a 3D array of instantaneous rates.

    Args:
        all_instantaneous_rates (list): time x nrows x ncols array containing
            instantaneous rates.
        start_time (float): Reference for the spike times (in ms)
        distribution (str): Distribution from which the probability of
            events are drawn.

    Returns:
        list: (nrows, ncols)-array of lists containing the times (in ms) at
            which a spike has been drawn.
            - The occurence or not of a spike is decided by the instantaneous
                rate and the distribution.
            - The function called to obtain the presence or absence of spike at
            each timestep with as argument the instantaneous rate is
            SPIKE_GENERATION_MAP[<distribution>].

    """
    _, nrows, ncols = all_instantaneous_rates.shape
    all_spike_times = np.empty((nrows, ncols), dtype=list)

    for row, col in itertools.product(range(nrows), range(ncols)):
        unit_instantaneous_rates = all_instantaneous_rates[:, row, col]
        all_spike_times[row, col] = unit_spike_times(
            unit_instantaneous_rates, start_time=start_time, distribution=distribution
        )

    return all_spike_times


def unit_spike_times(inst_rates, start_time=0.0, distribution="poisson"):
    """Draw spike times from a list of instantaneous rates.

    Args:
        inst_rates (list): List of rates (float values, expressed in Hz) for
            each timestep.
        start_time (float): Reference for the spike times (in ms)
        distribution (str): Distribution from which the probability of
            events are drawn.

    Returns:
        list: List of floats containing the times (in ms) at which a spike has
            been drawn.
            - The occurence or not of a spike is decided by the instantaneous
                rate and the distribution.
            - The function called to obtain the presence or absence of spike at
            each timestep with as argument the instantaneous rate is
            SPIKE_GENERATION_MAP[<distribution>].

    """
    spike_times = []
    draw_func = SPIKE_GENERATION_MAP[distribution]  # <- draw_poisson

    for t, rate in enumerate(inst_rates):
        if draw_func(rate):
            # Add 1 as NEST accepts only strictly positive spike times.
            spike_times.append(t + start_time + 1.0)
    return spike_times


def draw_poisson(rate, dt=0.001):
    """Return probability of having at list one spike during an interval.

    NB: This is an approximation to the actual poisson distribution as we
    collapse the probability of having any stricly positive number of events
    during an interval.

    Args:
        rate: instantaneous rate (in Hz)
        dt: length of the interval (default 1ms = 0.001s as the rate is in
            Hertz)

    """
    return np.random.poisson(float(rate * dt)) > 0


SPIKE_GENERATION_MAP = {"poisson": draw_poisson}
