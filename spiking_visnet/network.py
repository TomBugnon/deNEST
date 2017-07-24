#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

'''Provides the `Network` class.'''

import collections
import pprint

from .nestify.format_net import get_network
from .nestify.init_nest import init_nest
from .utils.structures import dictify

# TODO: move functionality from nestify/format_net to this class


class Network(collections.UserDict):

    def __init__(self, params):
        super().__init__(get_network(params))

    def init_nest(self, kernel_params):
        """ init_network updates in place the network dictionaries to add the
        gid's of the created nest objects """
        init_nest(self, kernel_params)

    def input_layer(self):
        # Returns (name, nest_params, params) for an arbitrary input layer
        name = self['areas']['input_area'][0]
        return (name,
                self['layers'][name]['nest_params'],
                self['layers'][name]['params'])

    def filters(self):
        """ Returns the 'filters' dictionary for an arbitrary input layer, or -1
        if there is no filter
        """
        (_, _, params) = self.input_layer()
        return params.get('filters', -1)

    def input_res(self):
        (_, nest_params, _) = self.input_layer()
        return (nest_params['columns'], nest_params['rows'])
