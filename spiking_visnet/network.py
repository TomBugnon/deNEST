#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""Provides the `Network` class."""

import collections

from .nestify.format_net import get_network
from .nestify.init_nest import init_nest

# TODO: move functionality from nestify/format_net to this class


class Network(collections.UserDict):
    """Network class."""

    def __init__(self, params):
        """Initialize object from parameter tree."""
        super().__init__(get_network(params))

    def init_nest(self, kernel_params):
        """Initialize NEST kernel and network.

        The GIDs of the created NEST objects are added in place to the Network
        object.
        """
        init_nest(self, kernel_params)

    def input_layer(self):
        """Return (name, nest_params, params) for an arbitrary input layer."""
        name = self['areas']['input_area'][0]
        return (name,
                self['layers'][name]['nest_params'],
                self['layers'][name]['params'])

    def filters(self):
        """Return the 'filters' dictionary for an arbitrary input layer.

        If no filter is applied to the input layers, return -1
        """
        (_, _, params) = self.input_layer()
        return params.get('filters', -1)

    def input_res(self):
        """Return the resolution of the network's input layers."""
        (_, nest_params, _) = self.input_layer()
        return (nest_params['columns'], nest_params['rows'])
