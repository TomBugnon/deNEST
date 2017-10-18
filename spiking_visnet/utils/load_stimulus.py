#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

from os.path import basename, isdir, isfile, join

import numpy as np

from ..save import load_as_numpy, load_yaml
from ..user_config import INPUT_SUBDIRS, METADATA_FILENAME


def load_raw_stimulus(input_path, session_stim_yaml):
    """Load the stimulus for the session.

    - If ``input_path`` points to a numpy array, load and return it.
    - if the ``input_path`` points to a directory, use it as the formatted
    directory in which to search for the session's stimulus yaml file

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
    # Single movie given by user.
    if isfile(input_path):
        print(f'-> Loading input from numpy at `{input_path}`')
        movie = load_as_numpy(input_path)
        # All frames have the same filename
        labels = [basename(input_path)
                  for i in range(np.size(movie, 0))]
        return (movie, labels, None)
    # Concatenate multiple movies from stimulus yaml file
    print(f'Loading input from input directory: `{input_path}`,'
          f' using stimulus yaml `{session_stim_yaml}`')
    return load_stim_from_yaml(input_path, session_stim_yaml)


def load_stim_from_yaml(input_dir, session_stims_filename):
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
    all_movies = []
    labels = []
    for movie_filename in stimulus_params['sequence']:
        # Load movie
        movie = load_as_numpy(join(input_dir,
                                   INPUT_SUBDIRS['preprocessed_input_sets'],
                                   stimulus_params['set_name'],
                                   movie_filename))
        all_movies.append(movie)
        # Save filename for each frame
        labels += [movie_filename for i in range(np.size(movie, 0))]

    # Check that we loaded something
    assert(all_movies), "Could not load any stimuli"
    # Check that all movies have same number of dimensions.
    assert(len(set([np.ndim(movie) for movie in all_movies])) == 1), \
           'Not all loaded movies have same dimensions'

    # Load metadata
    metadata = load_yaml(stimulus_params['set_name'], METADATA_FILENAME)

    return (np.concatenate(all_movies, axis=0), labels, metadata)
