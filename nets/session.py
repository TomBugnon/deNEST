#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py
"""Represent a sequence of stimuli."""

import time
from pprint import pformat

import numpy as np
from tqdm import tqdm

from . import save
from .utils.load_stimulus import load_raw_stimulus
from .utils.misc import pretty_time

# pylint:disable=missing-docstring

# Simulation time if self.params['max_session_sim_time'] == float('inf')
MAX_SIM_TIME_NO_INPUT = 10000.

class Session:
    """Represents a sequence of stimuli."""

    def __init__(self, name, params, start_time=0):
        print(f'-> Creating session `{name}`')
        self.name = name
        self.params = params
        # Initialize the session start and end times
        self._start = start_time
        assert 'simulation_time' in self.params
        self._simulation_time = int(self.params['simulation_time'])
        assert self._simulation_time > 0, ("Session's simulation time should be"
                                           " strictly positive (NEST kernel bug"
                                           " otherwise)")
        self._end = self._start + self._simulation_time
        # Initialize _stim dictionary
        self._stimulus = None
        # Whether we inactivate all recorders
        self._record = self.params.get('record', True)

    @property
    def end(self):
        return self._end

    @property
    def start(self):
        return self._start

    def __repr__(self):
        return '{classname}({name}, {params})'.format(
            classname=type(self).__name__,
            name=self.name,
            params=pformat(self.params))

    def initialize(self, network):
        """Initialize session.

        1- Reset Network
        2- Change network's dynamic variables.
        If there are InputLayers:
            3- Load stimuli
            5- Set input spike times or input rates.
        """
        # Reset network
        if self.params.get('reset_network', False):
            network.reset()

        # Change dynamic variables
        network.change_synapse_states(self.params.get('synapse_changes', []))
        network.change_unit_states(self.params.get('unit_changes', []))

        if network.any_inputlayer:
            # Load stimuli
            self._stimulus = self.load_stim(crop_shape=network.max_input_shape)
            # Set input spike times in the future.
            network.set_input(self.stimulus, start_time=self._start)

        # Inactivate all the recorders and connection_recorders for
        # `self._simulation_time`
        if not self._record:
            self.inactivate_recorders(network)

    def inactivate_recorders(self, network):
        """Set 'start' of all (connection_)recorders at the end of session."""
        # TODO: We need to do this differently if we start playing with the
        # `origin` flag of recorders, eg to repeat experiments. Hence the
        # safeguard:
        import nest
        for recorder in network.get_recorders():
            assert nest.GetStatus(recorder.gid, 'origin')[0] == 0.
        # Verbose
        print(f'Inactivating all recorders for session {self.name}:')
        # Set start time in the future
        network.recorder_call(
            'set_status',
            {'start': nest.GetKernelStatus('time') + self._simulation_time}
        )

    def run(self, network):
        """Initialize and run session."""
        import nest
        assert self.start == int(nest.GetKernelStatus('time'))
        print("Initialize session...")
        self.initialize(network)
        print("done...\n")
        print(f"Running session `{self.name}` for `{self.simulation_time}`ms")
        start_real_time = time.time()
        nest.Simulate(self.simulation_time)
        print(f"done.")
        print(f"Session `{self.name}` virtual running time: "
              f"`{self.simulation_time}`ms")
        print(f"Session `{self.name}` real running time: "
              f"{pretty_time(start_real_time)}...\n")
        assert self.end == int(nest.GetKernelStatus('time'))

    def save_metadata(self, output_dir):
        """Save session metadata (stimuli, ...)."""
        if self.params.get('save_stim', True) and self._stimulus is not None:
            save.save_array(save.output_path(output_dir,
                                             'session_movie',
                                             session_name=self.name),
                            self.stimulus['movie'])
            save.save_array(save.output_path(output_dir,
                                             'session_labels',
                                             session_name=self.name),
                            self.stimulus['labels'])
            save.save_as_yaml(save.output_path(output_dir,
                                               'session_metadata',
                                               session_name=self.name),
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

        The stimulus movie loaded from file is scaled by the session parameter
        `input_rate_scale_factor` (default 1.0). The session stimulus movie
        will be further scaled by the Network-wide Layer parameter
        `input_rate_scale_factor`
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

        # Scale the raw input by the session's scaling factor.
        scale_factor = self.params.get('input_rate_scale_factor', 1.0)
        raw_movie = raw_movie * scale_factor
        print(f'--> Apply session input_rate_scale_factor: {scale_factor}')

        # Expand from frame to timesteps
        labels = expand_raw_stimulus(raw_labels,
                                self.params.get('time_per_frame', 1.),
                                self.simulation_time)
        movie = expand_raw_stimulus(raw_movie,
                               self.params.get('time_per_frame', 1.),
                               self.simulation_time)

        return {'movie': movie,
                'labels': labels,
                'metadata': metadata}


def expand_raw_stimulus(list_or_array, nrepeats, target_length):
    """Repeat elems along the first dimension and adjust length to target.

    We first expand the array by repeating each element `nrepeats` times, and
    then adjust to the target length by either trimming or repeating the whole
    array.
    """
    extended_arr = np.repeat(list_or_array, nrepeats, axis=0)
    n_rep, remainder = divmod(target_length, len(extended_arr))
    return np.concatenate(
        [extended_arr for i in range(n_rep)] \
        + [extended_arr[:remainder]],
        axis=0
    )
