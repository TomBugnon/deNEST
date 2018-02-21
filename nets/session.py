#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py
"""Represent a sequence of stimuli."""


import time
from os.path import join
from pprint import pformat

import numpy as np

from . import save
from .utils.load_stimulus import load_raw_stimulus
from .utils.misc import pretty_time


class Session:
    """Represents a sequence of stimuli."""

    def __init__(self, name, params):
        print(f'-> Creating session `{name}`')
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
        self._stimulus = self.load_stim(crop_shape=network.max_input_shape)
        self._simulation_time = float(np.size(self.stimulus['movie'], axis=0))

        # Reset network
        if self.params.get('reset_network', False):
            network.reset()

        # Change dynamic variables
        network.change_synapse_states(self.params.get('synapse_changes', []))
        network.change_unit_states(self.params.get('unit_changes', []))

        # Set input spike times in the future.
        network.set_input(self.stimulus, start_time=self._start)

    def run(self, network):
        """Initialize and run session."""
        import nest
        self._start = int(nest.GetKernelStatus('time'))
        print("Initialize session...")
        self.initialize(network)
        print("done...\n")
        print(f"Running session `{self.name}` for `{self.simulation_time}`ms")
        start_time = time.time()
        nest.Simulate(self.simulation_time)
        print(f"done.")
        print(f"Session `{self.name}` virtual running time: "
              f"`{self.simulation_time}`ms")
        print(f"Session `{self.name}` real running time: "
              f"{pretty_time(start_time)}...\n")
        self._end = int(nest.GetKernelStatus('time'))

    def save(self, output_dir):
        """Save full stim (per timestep), labels (per timestep) and metadata."""
        if self.params.get('save_stim', True) and self._stimulus is not None:
            save.save_array(save.output_path(output_dir, 'movie', self.name),
                            self.stimulus['movie'])
            save.save_array(save.output_path(output_dir, 'labels', self.name),
                            self.stimulus['labels'])
            save.save_as_yaml(save.output_path(output_dir, 'metadata',
                                               self.name),
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
        """Load and return the session's input movie.

        See README.md and `load_raw_stimulus` function about how the input is
        loaded from the `input_path` simulation parameter and the
        `session_input` session parameter.
        """
        # Input path can be either to a file or to the structured input dir
        input_path = self.params['input_path']
        session_input = self.params['session_input']
        (raw_movie,
         raw_labels,
         metadata) = load_raw_stimulus(input_path, session_input)

        # Crop to adjust to network's input layer shape
        if crop_shape is not None:
            raw_movie = raw_movie[:, :, :crop_shape[0], :crop_shape[1]]

        # Expand from frame to timesteps
        labels = frames_to_time(raw_labels, self.params.get('time_per_frame',
                                                            1.))
        movie = frames_to_time(raw_movie, self.params.get('time_per_frame', 1.))

        simulation_time = int(min(np.size(movie, axis=0),
                                  self.params.get('max_session_sim_time',
                                                  float('inf'))))

        return {'movie': movie[:simulation_time],
                'labels': labels[:simulation_time],
                'metadata': metadata}


def frames_to_time(list_or_array, nrepeats):
    """Repeat elements along the first dimension."""
    return np.repeat(list_or_array, nrepeats, axis=0)
