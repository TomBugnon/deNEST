#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess.py

import os
from os.path import exists, isdir, isfile, join

import numpy as np
from tqdm import tqdm

from . import downsample, filt, normalize
from ..utils.structures import mkdir_ifnot

PREPROCESS_MAPPING = {
    'downsample': downsample.downsample,
    'filter': filt.filter,
    'normalize': normalize.normalize
}

METADATA_FILENAME = 'metadata.yml'


def preprocess_all(input_dir, prepro_subdir_str, network, prepro_params):

    # Preprocess files

    # Get dirs
    raw_dir = join(input_dir, 'raw_input')
    prepro_dir = join(input_dir, 'preprocessed_input', prepro_subdir_str)

    # Make dir
    mkdir_ifnot(prepro_dir)

    # Get files to be processed
    all_raw_files = [f for f in os.listdir(raw_dir)
                     if is_input_file(raw_dir, f)]
    all_prepro_files = [f for f in os.listdir(prepro_dir)
                        if is_input_file(prepro_dir, f)]
    todo_files = [f for f in all_raw_files if f not in all_prepro_files]

    for f in tqdm(todo_files,
                  desc=('Preprocess ' + str(len(todo_files)) + ' files')):

        movie = np.load(join(raw_dir, f))
        np.save(join(prepro_dir, f),
                preprocess(movie, network, prepro_params))

    # Create metadata file
    create_metadata(prepro_dir, prepro_params, network)

    # Create file sets for the new preprocessing pipeline
    update_sets(input_dir, prepro_subdir_str, prepro_dir)

    print('... done.')


def update_sets(input_dir, prepro_subdir_str, prepro_dir):
    """ For all the input subsets in raw_input_sets, use symlinks to create the
    equivalent preprocessed subset for the pipeling described by
    <prepro_subdir_str>.
    """

    print('Update sets')
    setdir = join(input_dir, 'raw_input_sets')

    all_setnames = [setname for setname in os.listdir(setdir)
                    if isdir(join(setdir, setname))]

    # Create sets of files + metadata
    for setname in all_setnames:
        create_set(input_dir,
                   setname,
                   prepro_subdir_str,
                   prepro_dir)


def create_set(input_dir, setname, prepro_subdir_str, prepro_dir):

    raw_set_dir = join(input_dir, 'raw_input_sets', setname)
    prepro_set_dir = join(input_dir, 'preprocessed_input_sets',
                          preprocessed_set_name(setname, prepro_subdir_str))

    # Create prepro_set dir
    mkdir_ifnot(prepro_set_dir)

    # Create simlinks to files if they don't already exist

    for f in [f for f in os.listdir(raw_set_dir)
              if not f == METADATA_FILENAME
              and not exists(join(prepro_set_dir, f))]:
        print(f)
        os.symlink(join(prepro_dir, f),
                   join(prepro_set_dir, f))

    # Copy preprocessing metadata into set directory
    # print(exists(join(prepro_set_dir, METADATA_FILENAME)))
    if not isfile(join(prepro_set_dir, METADATA_FILENAME)):
        os.symlink(join(prepro_dir, METADATA_FILENAME),
                   join(prepro_set_dir, METADATA_FILENAME))


def preprocessed_set_name(raw_set_name, prepro_subdir_str):
    return raw_set_name + '_' + prepro_subdir_str


def is_input_file(dirpath, filename):
    return (isfile(join(dirpath, filename))
            and not filename == METADATA_FILENAME)


def preprocess(movie, network, prepro_params):
    for step in prepro_params['preprocessing']:
        movie = PREPROCESS_MAPPING[step](movie, prepro_params, network)
    return movie


# TODO
def create_metadata(savedir, preprocessing_params, network):

    open(join(savedir, METADATA_FILENAME), 'a').close()
