#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# normalize.py
"""Perform contrast-normalization of input image."""

import numpy as np


# TODO
def normalize(input_movie, preprocessing_params, network):
    """Normalize movie contrast."""
    Warning('Are you sure you want to normalize by frame?')
    return input_movie / np.max(input_movie)


def get_string(*_):
    """Return summary string of this preprocessing step."""
    return 'contrastnorm_'
