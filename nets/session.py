#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py
"""Represent a sequence of stimuli."""

import time
from pprint import pformat

import numpy as np

from .base_object import ParamObject
from .utils.load_stimulus import load_raw_stimulus
from .utils.misc import pretty_time
from .utils.validation import ParameterError

# pylint:disable=missing-docstring


class Session(ParamObject):
    """Represents a sequence of stimuli.

    Args:
        name (str): Name of the session
        params (dict-like): Dictionary specifying session parameters. The
            following keys are recognized:
                - ``simulation_time`` (float): Duration of the session in msec.
                    (mandatory)
                - ``reset_network`` (bool): If true, ``nest.ResetNetwork()`` is
                    called during session initialization (default ``False``)
                - ``record`` (bool): If false, the ``start_time`` field of
                    recorder nodes in NEST is set to the end time of the
                    session, so that no data is recorded during the session
                    (default ``True``)
                - ``unit_changes`` (list): List describing the changes applied
                    to certain units before the start of the session.
                    Passed to ``Network.change_unit_states``. Refer to that
                    method for a description of how ``unit_changes`` is
                    formatted and interpreted. No changes happen if empty.
                    (default [])
                - ``synapse_changes`` (list): List describing the changes
                    applied to certain synapses before the start of the session.
                    Passed to ``Network.change_synapse_changes``. Refer to that
                    method for a description of how ``synapse_changes`` is
                    formatted and interpreted. No changes happen if empty.
                    (default [])
                - ``inputs`` (list): List describing the input applied to each
                    of the network's ``InputLayer`` objects during the session.
                    Refer to ``Session.load_input_array`` and
                    ``InputLayer.set_input`` for a description of how the inputs
                    are loaded and converted to stimulator activity. (default
                    [])

    Kwargs:
        start_time (float): Time of kernel in msec when the session starts
            running.
        input_path (str): Path to an input file or to the directory in which
            input files are searched for for each session. If ``input_path``
            points towards a loadable numpy input array, it will be used for
            setting the `InputLayer` layers' input. Otherwise, ``input_path`` is
            interpreted as a directory in which input array files are searched.
    """

    # Validation of `params`
    RESERVED_PARAMS = None
    MANDATORY_PARAMS = ['simulation_time']
    OPTIONAL_PARAMS = {
        'reset_network': False,
        'record': True,
        'unit_changes': [],
        'synapse_changes': [],
        'inputs': []
    }

    def __init__(self, name, params, start_time=0, input_path=None):
        print(f'-> Creating session `{name}`')
        # Sets self.name / self.params  and validates params
        super().__init__(name, params)
        self.input_path = input_path
        # Initialize the session start and end times
        self._start = start_time
        self._simulation_time = int(self.params['simulation_time'])
        if not self._simulation_time > 0:
            raise ParameterError(
                f"Parameter `simulation_time` of session {name} should be"
                f"strictly positive."
            )
        self._end = self._start + self._simulation_time
        # Initialize input arrays
        self._input_arrays = None

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

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
        # Reset network
        if self.params['reset_network']:
            network.reset()

        # Change dynamic variables
        network.change_synapse_states(self.params['synapse_changes'])
        network.change_unit_states(self.params['unit_changes'])

        # Inactivate all the recorders and connection_recorders for
        # `self._simulation_time`
        if not self.params['record']:
            self.inactivate_recorders(network)

        # Set input for each inputlayer
        inputlayers = network._get_layers(layer_type='InputLayer')
        self._input_arrays = {}
        for inputlayer in inputlayers:
            if inputlayer.name not in self.params['inputs'].keys():
                raise ParameterError(
                    f"No input defined in session {self.name} for InputLayer"
                    f"{str(inputlayer)}. Please check the `inputs` session"
                    f"parameter"
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
        """Set 'start' of all (connection_)recorders at the end of session.

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
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

        # TODO: Validate params

        # Input path can be either to an input array or to the directory in
        # which input # arrays are searched
        file = input_params['file']
        raw_input_array = load_raw_stimulus(self.input_path, file)

        # Crop to adjust to network's input layer shape
        layer_shape = input_layer.shape  # (row, col)
        raw_input_array_rowcol = (raw_input_array.shape[1],
                                  raw_input_array.shape[2])  # (row, col)

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
