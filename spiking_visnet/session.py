#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py


"""Represent a sequence of stimuli."""

from os.path import join
from pprint import pformat

import numpy as np

from . import save
from .user_config import INPUT_DIR
from .utils.load_stimulus import load_raw_stimulus
from .utils.sparsify import save_array


class Session:
    """Represents a sequence of stimuli."""

    def __init__(self, name, params):
        print('create Session')
        self.name = name
        self.params = params
        # Initialize the session start and end times
        self._start = 0
        self._end = 0
        self._simulation_time = None
        # Initialize _stim dictionary
        self._stimulus = None

    def __repr__(self):
        return '{classname}({name}, {params})'.format(
            classname=type(self).__name__,
            name=self.name,
            params=pformat(self.params))

    def initialize(self, network):
        """Initialize session.

        1- Load stimuli
        2- Reset Network
        3- Change network's dynamic variables.
        4- Set input spike times or input rates.

        """
        # Load stimuli
        print("-> Load session's stimulus")
        self._stimulus = self.load_stim(crop_shape=network.max_input_shape)
        self._simulation_time = float(np.size(self.stimulus['movie'], axis=0))

        # Reset network
        if self.params.get('reset_network', False):
            network.reset()

        # Change dynamic variables
        # network.change_synapse_states(self.params.get('synapse_changes', []))
        network.change_unit_states(self.params.get('unit_changes', []))

        # Set input spike times in the future.
        network.set_input(self.stimulus, start_time=self._start)

    def run(self, network):
        """Initialize and run session."""
        import nest
        self._start = int(nest.GetKernelStatus('time'))
        print("Initialize session")
        self.initialize(network)
        print(f"Run `{self.simulation_time}`ms")
        nest.Simulate(self.simulation_time)
        self._end = int(nest.GetKernelStatus('time'))

    def save(self, output_dir):
        """Save full stim (per timestep), labels (per timestep) and metadata."""
        if self.params.get('save_stim', True) and self._stimulus is not None:
            save_array(join(output_dir,
                            save.movie_filename(self.name)),
                       self.stimulus['movie'])
            save_array(join(output_dir,
                            save.labels_filename(self.name)),
                       self.stimulus['labels'])
            save.save_as_yaml(join(output_dir,
                                   save.metadata_filename(self.name)),
                              self.stimulus['metadata'])

    @property
    def stimulus(self):
        return self._stimulus

    @property
    def duration(self):
        return range(self._start, self._end)

    @property
    def simulation_time(self):
        return self._simulation_time

    def load_stim(self, crop_shape=None):
        # Input path can be either to a file or to the structured input dir
        input_path = self.params.get('input', INPUT_DIR)
        session_stim_filename = self.params.get('session_stimulus', None)
        (raw_movie,
         raw_labels,
         metadata) = load_raw_stimulus(input_path, session_stim_filename)

        # Crop to adjust to network's input layer shape
        if crop_shape is not None:
            raw_movie = raw_movie[:, :, :crop_shape[0], :crop_shape[1]]

        # Expand from frame to timesteps
        labels = frames_to_time(raw_labels, self.params['time_per_frame'])
        movie = frames_to_time(raw_movie, self.params['time_per_frame'])

        simulation_time = int(min(np.size(movie, axis=0),
                                  self.params['max_session_sim_time']))

        return {'movie': movie[:simulation_time],
                'labels': labels[:simulation_time],
                'metadata': metadata}


def frames_to_time(list_or_array, nrepeats):
    """Repeat elements along the first dimension."""
    return np.repeat(list_or_array, nrepeats, axis=0)
