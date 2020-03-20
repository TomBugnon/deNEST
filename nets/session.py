#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py
"""Represent a sequence of stimuli."""

import time
from pprint import pformat

import numpy as np

from . import save
from .utils.load_stimulus import load_raw_stimulus
from .utils.misc import pretty_time

# pylint:disable=missing-docstring


class Session:
    """Represents a sequence of stimuli.

    Args:
        name: Session name.
        params: Session parameters.
    
    Kwargs:
        start_time: Time of kernel in seconds when session starts running.
    """

    def __init__(self, name, params, start_time=0):
        print(f'-> Creating session `{name}`')
        self.name = name
        self.params = params
        # Initialize the session start and end times
        self._start = start_time
        self._simulation_time = int(self.params['simulation_time'])
        if not self._simulation_time > 0:
            raise ValueError(
                f"Session parameter `simulation_time` should be strictly"
                f"positive."
            )
        self._end = self._start + self._simulation_time
        # Initialize input arrays
        self._input_arrays = None
        # Whether we inactivate all recorders
        self._record = self.params.get('record', True)

    @property
    def end(self):
        """Return kernel time at session's end."""
        return self._end

    @property
    def start(self):
        """Return kernel time at session's start."""
        return self._start

    @property
    def input_arrays(self):
        """Return ``{<input_layer>: <input_array>}`` dict."""
        return self._input_arrays

    def __repr__(self):
        return '{classname}({name}, {params})'.format(
            classname=type(self).__name__,
            name=self.name,
            params=pformat(self.params))

    def initialize(self, network):
        """Initialize session.

            1. Reset Network
            2. Change network's dynamic variables.
            3. (possibly) inactivate recorders
            4. For each InputLayer
                1. Load input array
                2. Set layer's spike times or input rates from input array
        """
        # Reset network
        if self.params.get('reset_network', False):
            network.reset()

        # Change dynamic variables
        network.change_synapse_states(self.params.get('synapse_changes', []))
        network.change_unit_states(self.params.get('unit_changes', []))

        # Inactivate all the recorders and connection_recorders for
        # `self._simulation_time`
        if not self._record:
            self.inactivate_recorders(network)

        # Set input for each inputlayer
        inputlayers = network._get_layers(layer_type='InputLayer')
        self._input_arrays = {}
        for inputlayer in inputlayers:
            if inputlayer.name not in self.params['inputs'].keys():
                raise ValueError(
                    f"No input defined for InputLayer {str(inputlayer)}"
                )

            print(f"Setting input for InputLayer `{inputlayer.name}`")

            # Load input array 
            input_array = self.load_input_array(
                inputlayer,
                self.params['inputs'][inputlayer.name]
            )
            self._input_arrays[inputlayer.name] = input_array

            # Set input spike times in the future.
            inputlayer.set_input(input_array, start_time=self._start)

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
        """Save session metadata (stimulus array, ...)."""
        pass

    @property
    def input_array(self):
        return self._input_array

    @property
    def duration(self):
        return range(self._start, self._end)

    @property
    def simulation_time(self):
        return self._simulation_time

    def load_input_array(self, input_layer, input_params):
        """Load and return the session's input array for an ``InputLayer``

        Args:
            input_layer (InputLayer): Layer of type 'InputLayer'
            input_params (dict): Dictionary specifying the input for this layer.
                One of the keys of the ``inputs`` session parameter. Should have
                the following form::
                    {
                        'file': <input_file>
                        'time_per_frame': <time_per_frame>
                        'rate_scaling_factor': <rate_scaling_factor>
                    }
                Where:
                    - <file> points to the input array used to set the
                        stimulator's firing rates. Refer to
                        `utils.load_stimulus` for a description of how the array
                        is loaded from this parameter and the `input_path`
                        simulation parameter
                    - <rate_scaling_factor> scales the input array's values
                    - <time_per_frame> is the time in msec during which each of
                        the input array's "frames" is shown to the network.
        """
        # Input path can be either to a file or to the structured input dir
        input_path = self.params['input_path']
        file = input_params['file']
        raw_input_array = load_raw_stimulus(input_path, file)

        # Crop to adjust to network's input layer shape
        layer_shape = input_layer.shape  # (row, col)
        raw_input_array_rowcol = (raw_input_array.shape[1],
                                  raw_input_array.shape[2]) # (row, col)
        
        if not np.all(layer_shape <= raw_input_array_rowcol):
            raise ValueError(
                f"Invalid shape for input array at `file` for layer"
                f"{input_layer} "
            )
        cropped_input_array = raw_input_array[
            :,  # time,
            :layer_shape[0], :layer_shape[1]  # row, col
        ]

        # Scale the raw input by the session's scaling factor.
        scale_factor = input_params.get('rate_scaling_factor', 1.0)
        scaled_input_array = cropped_input_array * scale_factor
        print(f'--> Apply scaling factor to array: {scale_factor}')

        # Expand from frame to timesteps
        input_array = expand_stimulus_array(
            scaled_input_array,
            input_params.get('time_per_frame', 1.),
            self.simulation_time
        )

        return input_array


def expand_stimulus_array(list_or_array, nrepeats, target_length):
    """Repeat elems along the first dimension and adjust length to target.

    We first expand the array by repeating each element `nrepeats` times, and
    then adjust to the target length by either trimming or repeating the whole
    array.
    """
    extended_arr = np.repeat(list_or_array, nrepeats, axis=0)
    n_rep, remainder = divmod(target_length, len(extended_arr))
    return np.concatenate(
        [extended_arr for i in range(n_rep)] + [extended_arr[:remainder]],
        axis=0
    )
