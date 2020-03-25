#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/nest_object.py
"""Base class for representations of NEST objects."""

import functools
from pprint import pformat
import copy as cp


@functools.total_ordering
class NestObject:
    """Base class for a named NEST object.

    Args:
        name (str): The name of the object.
        params (Params): The object parameters. Interpreted by NETS.
        nest_params (Params): The object parameters passed to NEST.

    Objects are ordered and hashed by name.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, name, params, nest_params):
        self.name = name
        # Flatten the parameters to a dictionary (and make a copy)
        self.params = cp.deepcopy(params)
        self.nest_params = cp.deepcopy(nest_params)
        # Whether the object has been created in NEST
        self._created = False

    # pylint: disable=unused-argument,invalid-name
    def _repr_pretty_(self, p, cycle):
        opener = '{classname}({name}, '.format(
            classname=type(self).__name__, name=self.name)
        closer = ')'
        with p.group(p.indentation, opener, closer):
            p.breakable()
            p.pretty(self.params)
            p.pretty(self.nest_params)

    # pylint: enable=unused-argument,invalid-name

    def __repr__(self):
        return '{classname}({name}, {params}, {nest_params})'.format(
            classname=type(self).__name__, name=self.name,
            params=pformat(self.params), nest_params=pformat(self.nest_params)
        )

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)
