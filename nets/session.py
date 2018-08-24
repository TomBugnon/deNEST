#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py
"""Represent a sequence of stimuli."""

import time
from pprint import pformat

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from . import save
from .utils.load_stimulus import load_raw_stimulus
from .utils.misc import pretty_time

# pylint:disable=missing-docstring

# Simulation time if self.params['max_session_sim_time'] == float('inf')
MAX_SIM_TIME_NO_INPUT = 10000.

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
        # Whether we inactivate all recorders
        self._record = self.params.get('record', True)

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
            4- Get simulation time from stimulus length and max simulation time
            5- Set input spike times or input rates.
        else:
            3- Set simulation time as 'max_session_sim_time' session parameter
                or MAX_SIM_TIME_NO_INPUT if 'max_session_sim_time' is not
                defined
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
            self._simulation_time = float(np.size(self.stimulus['movie'], axis=0))
            # Set input spike times in the future.
            network.set_input(self.stimulus, start_time=self._start)
        else:
            self._simulation_time = self.params['max_session_sim_time']
            if self._simulation_time == float('inf'):
                self._simulation_time = MAX_SIM_TIME_NO_INPUT

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

    def save_data(self, output_dir, network, sim_params):
        """Save network's activity and clear memory.

        1- Possibly formats recorders (possibly in parallel)
        2- Possibly creates raster plots (must be in series)
        3- Possibly clears memory

        Args:
            output_dir (str):
            network (Network object):
            sim_params (Params object): Simulation parameters.
        """
        # Get relevant params from sim_params
        format_recorders =  sim_params.get('format_recorders', False)
        clear_memory = sim_params.get('clear_memory', False)
        # We save the rasters only if we clear memory at the end of each
        # session. Otherwise we save them once at the end of the whole
        # simulation
        save_nest_rasters = sim_params.get('save_nest_rasters', True) and clear_memory

        # Possibly format the recorders
        if format_recorders:
            # Make kwargs dict containing simulation parameters
            sim_kwargs = {
                'parallel': sim_params.get('parallel', True),
                'n_jobs': sim_params.get('n_jobs', -1),
            }
            # Make kwargs dict containing session parameters (eventually passed
            # to `Recorder.save`)
            session_kwargs = {
                'session_name': self.name,
                'start_time': self._start,
                'end_time': self._end,
            }
            all_recorders = network.get_recorders(recorder_class=None)
            format_all_recorders(
                all_recorders, output_dir, sim_kwargs, session_kwargs)

        # Save the rasters for population recorders (must be in series)
        if save_nest_rasters:
            print('Saving rasters...')
            network.recorder_call('save_raster', output_dir, self.name,
                                  recorder_class='population')
        ##
        # Clear memory for all recorders
        if clear_memory:
            print('Clearing memory...')
            network.recorder_call('clear_memory', recorder_class=None)
            print('... done \n')

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


def worker(recorder, output_dir, **kwargs):
    recorder.save_formatted(output_dir, **kwargs)


# TODO: Format only if the session has been recorded and figure out a way to
# load the data properly
def format_all_recorders(all_recorders, output_dir, sim_kwargs, save_kwargs):
    # Format all recorders (population and connection), possibly using joblib
    args_list = [(recorder, output_dir)
                 for recorder in all_recorders]
    parallel = sim_kwargs['parallel']
    n_jobs = sim_kwargs['n_jobs']

    # Verbose
    msg = (f"Formatting {len(args_list)} population/connection recorders "
           f"{'using' if parallel else 'without'} joblib: \n"
           f"...")
    print(msg)

    if parallel:
        Parallel(n_jobs=n_jobs, verbose=100, batch_size=1)(
            delayed(worker)(*args, **save_kwargs) for args in args_list
        )
    else:
        for args in tqdm(args_list,
                         desc=''):
            worker(*args, **save_kwargs)

    print('...Done formatting recorders\n')
