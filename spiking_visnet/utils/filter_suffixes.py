#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/filter_suffixes.py
"""Map between filter descriptions and input layer names."""

import itertools


def get_expanded_names(base_layer_name, filters):
    """Generate list of expanded layer names.

    Args:
        base_layer_name (str): Original layer name
        filters (dict): Filter dictionary

    Returns:
        list: list of expanded names.

    """
    return ([
        base_layer_name + suffix for suffix in get_extension_suffixes(filters)
    ] or [base_layer_name])


def get_summary_string(filters):
    """Return a string summarizing the number of filters in each dimension."""
    return get_extension_suffixes(filters)[-1]


def get_extension_suffixes(filters):
    """Generate suffixes from the filter dictionary.

    Return a list of suffixes that will be appended to the input layer names
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
    if not filters:
        return suffixes
    for dim in [d for d in filters['dimensions']
                if filters['dimensions'][d] > 1]:
        suffixes.append([
            filters['suffixes'][dim] + str(i + 1)
            for i in range(filters['dimensions'][dim])
        ])
    return ['_' + string for string in combine_strings(suffixes)]


def combine_strings(strings, separator=''):
    """Return a list of the combinations of a list of lists of strings.

    Args:
        strings (list[list]): Strings to be combined.

    Returns:
        list: All combinations of strings from the different sequences.

    Example:
        >>> strings = [['a1','a2'], ['b1','b2','b3']]
        >>> list(combine_strings)
        ['a1_b1', 'a1_b2', 'a1_b3', 'a2_b1', 'a2_b2', 'a2_b3']
    """
    if not strings:
        return []
    yield from (separator.join(comb) for comb in itertools.product(*strings))
