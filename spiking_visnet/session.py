#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py


"""Represent a sequence of stimuli."""


import collections
from os.path import basename, isdir, isfile, join

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
        # Load the stimuli
        self.full_stim = None
        self.labels = None
        self.stim_metadata = None
        self.load_full_session_stim()
        # Simulation time depends on length of input movie.
        self.sim_time = min(np.size(self.full_stim, axis=0),
                            self['max_session_sim_time'])

    def initialize(self, network):
        """Initialize session.

        1- Reset Network
        2- Change network's dynamic variables.
        3- Set input spike times or input rates.

        """
        # Network resetting
        if self['reset_network']:
            nest.ResetNetwork()

        # Change dynamic variables
        toggle_dynamic_synapses(network,
                                on_off=self.get('dynamic_synapses', 1))

        # Set input spike times in the future.
        curr_time = nest.GetKernelStatus('time')
        set_stimulators_state(network,
                              self,
                              start_time=curr_time)


    def run(self, network):
        """Initialize and run session."""
        print("Initialize session")
        self.initialize(network)
        print(f"Run `{self.sim_time}`ms")
        nest.Simulate(self.sim_time)


    def load_full_session_stim(self):
        """Save as Session attributes the full session stimulus.

        Create attributes:
        self.full_movie (np.array): the full session movie, after expansion
            (from frame fo timesteps) and shuffling.
        self.labels (list of strings): The filename of the movie each image
            originates from (by timestep)
        self.stim_metadata (dict or None): The preprocessing metadata used for
            the stimuli (None if input movie is from CLI)
        """
        full_stim_by_frame, \
        frame_filenames, \
        stim_metadata = self.load_session_stim()

        self.stim_metadata = stim_metadata

        # Expand from frame to timestep
        timestep_labels = frames_to_time(frame_filenames,
                                         self['time_per_frame'])
        full_stim_by_timestep = frames_to_time(full_stim_by_frame,
                                               self['time_per_frame'])
        self.full_stim, self.labels = self.shuffle_stim(full_stim_by_timestep,
                                                        timestep_labels)



    def load_session_stim(self):
        """Load the stimulus for the session.

        - If there is a user-specified input path in the session parameters
        ('user_input' key) pointing to  a numpy array, load and return it.
        - if the user_specified path points to a directory, use it as the
        INPUT_DIR in which to search for the session's stimuli yaml file
        (`session_stims` key)
        - Otherwise, load from the yaml file in the INPUT_DIR/stimuli
        subdirectory (`session_stims` key).

        Return:
        (tuple): (<stim>, <filenames>, <stim_metadata> ) where:
            <stim > (np - array): (nframes * nfilters * nrows * ncols) array.
            <frame_filenames> (list): list of length nframes containing the
                filename of the movie each frame is taken from.
            <stim_metadata > (dict or None):
                None if stimulus is loaded directly from a numpy array.
                Metadata of the preprocessing pipeline (used to map input layers
                and filter dimensions) otherwise.

        """
        user_input = self.get('user_input', False)
        # Single movie given by user.
        if user_input and isfile(user_input):
            stim = load_as_numpy(user_input)
            # All frames have the same filename
            frame_filenames = [basename(user_input)
                               for i in range(np.size(stim, 0))]
            return (stim, frame_filenames, None)
        # Multiple movies in stimulus yaml file
        elif user_input and isdir(user_input):
            input_dir = user_input
        else:
            input_dir = INPUT_DIR
        return load_stim_yaml(input_dir, self['session_stims'])

    # TODO:
    def shuffle_stim(self, stim, labels):
        """Shuffle a (T*nframes*nrows*ncols) stimulus in space and time.

        Apply the time shuffling to the timestep labels as well"""
        return stim, labels


def load_stim_yaml(input_dir, session_stims_filename):
    """Load and concatenate a sequence of movie stimuli from a 'stim' yaml file.

    Load only files that exist and are of non - null size. Concatenate along the
    first dimension(time).

    Args:
        <session_stims_path> (str): Path to the session's stimulus file.

    Return:
        (tuple): (<stim> , <frame_filenames>, <stim_metadata> ) where:
            <stim> (np - array): (T * nfilters * nrows * ncols) array.
            <frame_filenames> (list): list of length T containing the filename
                of the movie each frame is taken from.
            <stim_metadata> (dict): Metadata from preprocessing pipeline. Used
                to map input layers and filter dimensions.
    """
    # Load stimulus yaml file for the session. Contains the set and a sequence
    # of filenames.
    stimulus_params = load_yaml(join(input_dir,
                                    INPUT_SUBDIRS['stimuli'],
                                    session_stims_filename))

    # Load all the movies in a list of arrays, while saving the label for each
    # frame
    all_movie_list = []
    frame_filenames = []
    for movie_filename in stimulus_params['sequence']:
        # Load movie
        movie = load_as_numpy(join(input_dir,
                                   INPUT_SUBDIRS['preprocessed_input_sets'],
                                   stimulus_params['set_name'],
                                   movie_filename))
        all_movie_list.append(movie)
        # Save filename for each frame
        frame_filenames += [movie_filename for i in range(np.size(movie, 0))]


    # Check that we loaded something
    assert(all_movie_list), "Could not load any stimuli"
    # Check that all movies have same number of dimensions.
    assert(len(set([np.ndim(movie) for movie in all_movie_list])) == 1), \
           'Not all loaded movies have same dimensions'

    # Load metadata
    metadata = load_yaml(stimulus_params['set_name'], METADATA_FILENAME)

    return (np.concatenate(all_movie_list, axis=0),
    frame_filenames, metadata)


def frames_to_time(list_or_array, nrepeats):
    """Repeat elements along the first dimension."""
    return np.repeat(list_or_array, nrepeats, axis=0)
