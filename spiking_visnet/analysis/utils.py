#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analysis/utils.py

"""Utility functions for analysis routines and structures."""

import os.path

from ..utils.sparsify import load_as_numpy


def load_activity(layer_name, pop_name, output_dir, variable='spikes'):
    """Return spiking activity for a population."""
    filename = layer_name + '_' + pop_name + '_' + variable + '.npy'
    path = os.path.join(output_dir, filename)

    return load_as_numpy(path)
