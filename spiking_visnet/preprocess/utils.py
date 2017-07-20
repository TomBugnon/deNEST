#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocess/utils.py

import importlib

from . import downsample, filt, normalize

NAME_MAPPING = {
    'downsample': downsample.get_string,
    'filter': filt.get_string,
    'normalize': normalize.get_string
}


def preprocessing_subdir(prepro_params, network):
    ''' Uses the get_string function in each of the preprocessing steps' modules
    to generate a string describing the preprocessing pipeline.'''
    subdir = ''
    for prepro_step in prepro_params['preprocessing']:
        # use the get_string function in each of the preprocessing modules
        subdir += NAME_MAPPING[prepro_step](prepro_params, network)
    return subdir.strip('_')
