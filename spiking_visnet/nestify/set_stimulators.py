#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/init_spike_times.py


"""Set the spike timings of input layers from stimuli."""

import nest
import numpy as np
from tqdm import tqdm

from ..utils.filters_layers import filter_index


def set_stimulators_state(network, session, start_time=0.):
    """Set in NEST the spike times for all input neurons during next session.

    For each layer:
    - if the stimulators are poisson generators, set their rate according
        to the first frame of the full input movie (session.full_stim)
    - if the stimulators are spike generators, set their spike times from
        a poisson distribution of varying instantaneous rate given by the
        session's full input movie (session.full_stim)

    Args:
        network (Network object): network
        session (Session object): Session object. Contains the full stimulus
            defining the instantaneous rates (by timestep)
        start_time (float): Current time of the NEST kernel. Used to set the
            spike times in the future.

    """
    # Iterate on stimulation device layers.
    input_stim_layers = network.input_stim_layers()
    for input_stim_layer in tqdm(input_stim_layers,
                                 desc="Set input layers' activity"):
        # 3D array describing the movie shown to a single layer.
        filt_index = filter_index(input_stim_layer,
                                  stim_metadata=session.stim_metadata)
        layer_movie = session.full_stim[:, filt_index, :, :]

        # Get layer's population name and stimulator type.
        population_name = network.populations(input_stim_layer)[0]
        stim_type = network.stimulator_type(population_name)

        if stim_type == 'poisson_generator':

            # Show only first frame to network
            layer_image = layer_movie[0, :, :]
            set_poisson_rates(network, layer_image,
                              input_stim_layer, population_name)

        elif stim_type == 'spike_generator':

            set_spike_times(network, layer_movie, input_stim_layer,
                            population_name, start_time)


def set_poisson_rates(network, image, layer, population):
    """Set the rates of a poisson generators layer from an image.

    The rate of unit at location (row, col) is set as:
        <max_input_rate≥ * image[row, col]
    where <max_input_rate> is a network-wide parameter accessed in the layer's
    parameter dictionary.

    """
    max_input_rate = network['layers'][layer]['params']['max_input_rate']
    gid_locs = network.locations[layer][population]
    # Set rate for all the units in the layer.
    for gid, location in gid_locs['location'].items():
        rate = float(max_input_rate * image[location])
        nest.SetStatus((gid,), {'rate': rate})


# TODO
def set_spike_times(network, movie, layer, population, start_time=0.,
                    distribution='poisson'):
    """Draws spikes and set an input population state for a future session."""
    max_input_rate = network['layers'][layer]['params']['max_input_rate']
    gid_locs = network.locations[layer][population]

    for gid, location in gid_locs['location'].items():
        # Rate of firing during each frame for a single unit
        inst_rates = movie[:, location[0], location[1]] * max_input_rate

        # Times of spiking for that unit
        spike_times = draw_spike_times(inst_rates,
                                       start_time=start_time,
                                       distribution=distribution)

        nest.SetStatus((gid,), {'spike_times': spike_times})


def draw_spike_times(inst_rates, start_time=0., distribution='poisson'):
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
            - The

    """
    spike_times = []
    draw_func = SPIKE_GENERATION_MAP[distribution]  # <- draw_poisson

    for t, rate in enumerate(inst_rates):
        if draw_func(rate):
            # Add 1 as NEST accepts only strictly positive spike times.
            spike_times.append(t + start_time + 1.)
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


SPIKE_GENERATION_MAP = {'poisson': draw_poisson}
