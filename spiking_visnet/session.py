#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py


"""Represent a sequence of stimuli."""


import collections
from os import stat
from os.path import exists, join

import nest
import numpy as np

from user_config import INPUT_DIR, INPUT_SUBDIRS, METADATA_FILENAME

from .nestify.nest_modifs import toggle_dynamic_synapses
from .nestify.set_stimulators import set_stimulators_state
from .save import load_yaml
from .utils.sparsify import load_as_numpy


class Session(collections.UserDict):
    """Represents a sequence of stimuli."""

    def __init__(self, session_params):
        """Create."""
        print('create Session')
        super().__init__(session_params)

    def initialize(self, network, default_sim_time=20.):
        """Initialize session.

        1- Reset Network
        2- Change network's dynamic variables.
        3- Set input spike times or input rates.

        """
        # Network resetting
        if self['reset_network']:
            nest.ResetNetwork()

        # Change dynamic variables
        toggle_dynamic_synapses(network, on_off=self['dynamic_synapses'])

        # Set input.
        curr_time = nest.GetKernelStatus('time')
        full_stim, stim_metadata = load_session_stim(self['session_stims'])

        stimulator_type = set_stimulators_state(network,
                                                full_stim,
                                                self,
                                                stim_metadata,
                                                start_time=curr_time)

        # Get simulation time
        if stimulator_type == 'poisson_generator':
            sim_time = default_sim_time
        elif stimulator_type == 'spike_generator':
            sim_time = self['time_per_frame'] * np.size(full_stim, axis=0)

        return min(sim_time, self['max_session_sim_time'])

    def run(self, network, default_simulation_time=0.):
        """Initialize and run session."""
        print("Initialize session")
        sim_time = self.initialize(network)
        print(f"Run `{sim_time}`ms")
        nest.Simulate(sim_time)


def load_session_stim(session_stims_filename):
    """Load and concatenate a sequence of movie stimuli.

    Load only files that exist and are of non-null size. Concatenate along the
    first dimension (time).

    Args:
        - <session_stims_path> (str): value of the 'session_stims' entry in the
            session parameters. Filename in INPUT_DIR/stimuli directory.

    Return:
        - (tuple): (<stim>, <stim_metadata>) where:
            <stim> (np-array): (T * nfilters * nrows * ncols) array.
            <stim_metadata> (dict): Metadata from preprocessing pipeline. Used
                to map input layers and filter dimensions.
    """
    # Load all existing non-empty movies
    stimuli_params = load_yaml(INPUT_DIR,
                               INPUT_SUBDIRS['stimuli'],
                               session_stims_filename)
    full_movie_paths = [join(INPUT_DIR,
                             INPUT_SUBDIRS['preprocessed_input_sets'],
                             stimuli_params['set_name'],
                             filename)
                        for filename in stimuli_params['sequence']]

    movie_list = [load_as_numpy(filepath)
                  for filepath in full_movie_paths
                  if exists(filepath) and not (stat(filepath).st_size == 0)]

    # Check that we loaded something
    assert(movie_list), "Could not load stimuli"
    # Check that all movies have same number of dimensions.
    assert(len(set([np.ndim(movie) for movie in movie_list])) == 1,
           'Not all loaded movies have same dimensions')

    # Load metadata
    metadata = load_yaml(stimuli_params['set_name'], METADATA_FILENAME)

    return (np.concatenate(movie_list, axis=0), metadata)
