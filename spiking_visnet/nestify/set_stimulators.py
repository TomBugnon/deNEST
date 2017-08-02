#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/init_spike_times.py


"""Set the spike timings of input layers from stimuli."""

from itertools import chain, repeat

import nest
import numpy as np

from tqdm import tqdm

from ..utils.filters_layers import filter_index


def set_stimulators_state(network, full_stim, session_params, stim_metadata,
                          start_time=0.):
    """Set in NEST the spike times for all input neurons during next session.

    1- Shuffle the (nframes * nfilters * nrows * ncols) array in space and time.
    2- For each layer:
        - if the stimulators are poisson generators, set their rate according
            to the first frame of the movie_1
        - if the stimulators are spike generators, set their spike times from
            a poisson distribution of varying instantaneous rate depending on
            which frame is shown to the network.

    Args:
        - <network> (Network object)
        - <full_stim> (np.array): (nframes * nfilters * nrows * ncols) numpy
            array
        - <session_params> (dict): parameters of the session
        - <stim_metadata> (dict): metadata of the input preprocessing (used to
            map between filter and input layer).
        - <start_time> (float): Current time of the NEST kernel. Used to set the
            spike times in the future.

    Returns:
        - (str): Type of stimulator. Used to set simulation time.
            - 'poisson_generator' (default)
            - 'spike_generator' if there is at least one layer of spike
                generators
    """
    # Initialize Output
    stimulator_type = None

    # Shuffle stimulus
    shuffled_stim = shuffle_stim(full_stim, session_params)

    # Iterate on stimulation device layers.
    input_stim_layers = network.input_stim_layers()

    for input_stim_layer in tqdm(input_stim_layers,
                                 desc="Set input layers' activity"):

        # 3D array describing the movie shown to a single layer.
        filt_index = filter_index(input_stim_layer, stim_metadata)
        layer_movie = shuffled_stim[:, filt_index, :, :]

        # Get layer's population name and stimulator type.
        population_name = network.populations(input_stim_layer)[0]
        stim_type = network.stimulator_type(population_name)

        if stim_type == 'poisson_generator':

            # Show only first frame to network
            layer_image = layer_movie[0, :, :]
            set_poisson_rates(network, layer_image,
                              input_stim_layer, population_name)

            if not stimulator_type == 'spike_generator':
                stimulator_type = stim_type

        elif stim_type == 'spike_generator':

            set_spike_times(network, layer_movie, input_stim_layer,
                            population_name, start_time,
                            session_params['time_per_frame'])

            stimulator_type = stim_type

    return stimulator_type


# TODO:
def shuffle_stim(stim, session_params):
    """Shuffle a (T*nframes*nrows*ncols) stimulus in space and time."""
    return stim


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
                    time_per_frame=1, distribution='poisson'):
    """TODO."""
    max_input_rate = network['layers'][layer]['params']['max_input_rate']
    gid_locs = network.locations[layer][population]
    nframes, nrows, ncols = np.shape(movie)

    for gid, location in gid_locs['location'].items():
        # Rate of firing during each frame for a single unit
        frame_rates = movie[:, location[0], location[1]] * max_input_rate

        # Instantaneous rate of firing (a frame lasts time_per_frame ms.)
        inst_rates = list(chain.from_iterable(repeat(frame_rate,
                                                     int(time_per_frame))
                                              for frame_rate in frame_rates))
        # Times of spiking for that unit
        spike_times = draw_spike_times(inst_rates,
                                       start_time=start_time,
                                       distribution='poisson')

        nest.SetStatus((gid,), {'spike_times': spike_times})
    pass


def draw_spike_times(inst_rates, start_time=0., distribution='poisson'):
    """Draw spike times from a list of instantaneous rates.

    Args:
        - <inst_rates> (list): List of rates (float values, expressed in Hz) for
            each timestep.
        - <start_time> (float): Reference for the spike times (in ms)
        - <distribution> (str): Distribution from which the probability of
            events are drawn.

    Return:
        (list): List of floats containing the times (in ms) at which a spike has
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
        - <rate>: instantaneous rate (in Hz)
        - <dt>: length of the interval (default 1ms = 0.001s as the rate is in
            Hertz)

    """
    return (np.random.poisson(float(rate * dt)) > 0)


SPIKE_GENERATION_MAP = {'poisson': draw_poisson}
