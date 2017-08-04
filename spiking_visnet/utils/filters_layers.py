#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/filters_layers.py

"""Map between filters used in preprocessing and input layers of a network."""


# TODO
def filter_index(input_layer_name, stim_metadata=None):
    """Return the index of the filter associated to an input layer.

    If no preprocessing metadata is provided, return 0 (all input layers see
    the same image.)
    Otherwise, obtain the filter index from the preprocessing metadata.

    """
    if not stim_metadata:
        return 0
    else:
        # TODO
        return 0
