#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/filter_suffixes.py

import itertools
from collections import ChainMap


def get_expanded_names(base_layer_name, filters):
    '''
    Args:
        - <base_layer_name> (str)
        - <filters> (dict)
    '''

    return [
        base_layer_name + suffix for suffix in get_extension_suffixes(filters)
    ]


def get_extension_suffixes(filters):
    """Returns a list of suffixes that will be appended to the input layer names
    to describe which combination of filter dimensions each layer corresponds
    to.
    - All potential suffixes start with an underscore.
    - Degenerate dimensions are omitted.
    - If there is only one filter type overall, returns a singleton list
      containing an empty string.
    - If there is eg: 2 spatial frequencies, 2 orientations, 1 sign, returns
    - ['_sf1o1', '_sf1o2', '_sf2o1', '_sf2o2']
    """
    suffixes = []
    for dim in [d for d in filters['dimensions']
                if filters['dimensions'][d] > 1]:
        suffixes.append([filters['suffixes'][dim] + str(i + 1)
                         for i in range(filters['dimensions'][dim])])
    return (['_' + s for s in combine_strings(suffixes)]
            if suffixes else [''])


def combine_strings(s):
    """ Returns a list of the combination of strings in s.

    Args:
        - <s>: List of lists of strings, eg [['a1','a2'], ['b1','b2','b3'],...]
    Returns:
        <list>: List of the combinations (sets) of strings from the different
            sublists, with one string from each sublist, eg
                ['a1b1', 'a1b2', 'a1b3', 'a2b1',...]
    """
    return [''.join(tup) for tup in list(itertools.product(*s))]
