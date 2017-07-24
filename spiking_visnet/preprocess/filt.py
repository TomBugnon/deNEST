#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filt.py

from ..utils.filter_suffixes import get_summary_string


# TODO
def filter(input_movie, preprocessing_params, network):
    """Filter input_movie with set of filters described in network."""
    return input_movie


# TODO
def get_string(_, network):
    """Return summary string of this preprocessing step."""
    return 'filter' + get_summary_string(network.filters())
