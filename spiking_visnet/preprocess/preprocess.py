#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess.py

"""Preprocess all raw movies with a given pipeline."""

import os
import warnings
from os import makedirs
from os.path import exists, isdir, isfile, join

import numpy as np
from tqdm import tqdm

from user_config import INPUT_SUBDIRS, METADATA_FILENAME

from . import filt, normalize, resize
from ..save import save_as_yaml
from ..utils.sparsify import load_as_numpy, save_array

PREPROCESS_MAPPING = {
    'resize': resize.resize,
    'filter': filt.filter_movie,
    'normalize': normalize.normalize
}


def preprocess_all(input_dir, prepro_subdir_str, network, prepro_params):
    """Preprocess all raw movies and create the corresponding sets.

    Args:
        input_dir (str): Path to USER's input directory.
        prepro_subdir_str (str): String describing preprocessing pipeline.
        network (Network): Network this preprocessing is effected for.
        prepro_params (dict): Preprocessing parameter tree.

    """
    # Preprocess files

    # Get dirs
    raw_dir = join(input_dir, INPUT_SUBDIRS['raw_input'])
    prepro_dir = join(input_dir, INPUT_SUBDIRS['preprocessed_input'],
                      prepro_subdir_str)

    # Make preprocessed_input dir for this pipeline
    makedirs(prepro_dir, exist_ok=True)

    # Get files to be processed
    all_raw_files = [f for f in os.listdir(raw_dir)
                     if is_input_file(raw_dir, f)]
    all_prepro_files = [f for f in os.listdir(prepro_dir)
                        if is_input_file(prepro_dir, f)]
    todo_files = [f for f in all_raw_files if f not in all_prepro_files]

    # Preprocess and save each file
    for filename in tqdm(todo_files,
                  desc=('Preprocess ' + str(len(todo_files)) + ' files')):

        movie = load_as_numpy(join(raw_dir, filename))
        save_array(join(prepro_dir, filename),
                       preprocess(movie, network, prepro_params))

    # Create metadata file for this preprocessing pipeline
    create_metadata(prepro_dir, prepro_params, network)

    # Create file sets for this preprocessing pipeline
    update_sets(input_dir, prepro_subdir_str)

    # Create a default stimulus file for this preprocessing pipeline
    create_default_stim_yaml(input_dir, prepro_subdir_str)

    print('... done.')


def create_default_stim_yaml(input_dir, prepro_subdir_str):
    """Create a default stimulus yaml for the new preprocessing pipeline.

    The created default stimulus yaml points towards the newly preprocessed set
    named DEFAULT_SET (= 'set_1'). The stimulus sequence is the list of unique
    files in the default set.
    The file created by this function in <input_dir>/stimuli has name:
        DEFAULT_STIM_NAME+DEFAULT_SET+<prepro_subdir_str>+'.yml'
    and is of the form:
    "
    - set_name: <DEFAULT_SET> + 'prepro_subdir_str'
    - sequence:
        - filename1
        - filename2
        ...
    "

    """
    DEFAULT_SET = 'set_df'
    DEFAULT_STIM_NAME = 'stim_df'

    default_stim_filename = (DEFAULT_STIM_NAME + '_' + DEFAULT_SET + '_'
                             + prepro_subdir_str + '.yml')
    stim_dir_path = join(input_dir, INPUT_SUBDIRS['stimuli'])
    default_stim_path = join(stim_dir_path,
                             default_stim_filename)

    default_set_name = (DEFAULT_SET + '_' + prepro_subdir_str)
    default_set_path = join(input_dir,
                            INPUT_SUBDIRS['preprocessed_input_sets'],
                            default_set_name)

    # Check existence of a default set
    if not isdir(default_set_path):
        warnings.warn('No default set. Could not create default stimulus.')
    # Create the default stimulus yaml
    else:
        set_filenames = [f for f in os.listdir(default_set_path)
                         if not f == METADATA_FILENAME]

        makedirs(stim_dir_path, exist_ok=True)
        save_as_yaml(default_stim_path,
                     {
                         'sequence': set_filenames,
                         'set_name': default_set_name
                     })


def update_sets(input_dir, prepro_subdir_str):
    """Create preprocessed input sets.

    For all the input subsets in raw_input_sets, use symlinks to create the
    equivalent preprocessed subset for the pipeling described by
    <prepro_subdir_str>.

    """
    print('Update sets')
    setdir = join(input_dir, INPUT_SUBDIRS['raw_input_sets'])

    all_setnames = [setname for setname in os.listdir(setdir)
                    if isdir(join(setdir, setname))]

    # Create sets of files + metadata
    for setname in all_setnames:
        create_set(input_dir,
                   setname,
                   prepro_subdir_str)


def create_set(input_dir, setname, prepro_subdir_str):
    """Create a given preprocessed input subset from a raw input set."""
    raw_set_dir = join(input_dir, INPUT_SUBDIRS['raw_input_sets'], setname)
    prepro_set_dir = join(input_dir, INPUT_SUBDIRS['preprocessed_input_sets'],
                          preprocessed_set_name(setname, prepro_subdir_str))

    # Create prepro_set dir
    makedirs(prepro_set_dir, exist_ok=True)

    # Create simlinks to files if they don't already exist
    # Symlinks need to be relative
    for f in [f for f in os.listdir(raw_set_dir)
              if not f == METADATA_FILENAME
              and not exists(join(prepro_set_dir, f))]:
        print(f)
        os.symlink(join('..', '..', INPUT_SUBDIRS['preprocessed_input'],
                        prepro_subdir_str, f),
                   join(prepro_set_dir, f))

    # Copy preprocessing metadata into set directory
    # print(exists(join(prepro_set_dir, METADATA_FILENAME)))
    if not isfile(join(prepro_set_dir, METADATA_FILENAME)):
        os.symlink(join('..', '..', INPUT_SUBDIRS['preprocessed_input'],
                        prepro_subdir_str, METADATA_FILENAME),
                   join(prepro_set_dir, METADATA_FILENAME))


def preprocessed_set_name(raw_set_name, prepro_subdir_str):
    """Create string describing a preprocessed input set."""
    return raw_set_name + '_' + prepro_subdir_str


def is_input_file(dirpath, filename):
    """Discriminate between input files and metadata files."""
    return (isfile(join(dirpath, filename))
            and not filename == METADATA_FILENAME)


def preprocess(movie, network, prepro_params):
    """Apply preprocessing pipeline to a movie."""
    for step in prepro_params['preprocessing']:
        movie = PREPROCESS_MAPPING[step](movie, prepro_params, network)
    return movie


# TODO
def create_metadata(savedir, preprocessing_params, network):
    """Create metadata file describing full pipeline."""
    open(join(savedir, METADATA_FILENAME), 'a').close()
