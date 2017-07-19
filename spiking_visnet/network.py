#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

'''Provides the `Network` class.'''

import collections
import pprint

from .nestify.format_net import get_network
from .utils.structures import dictify

# TODO: move functionality from nestify/format_net to this class


class Network(collections.UserDict):

    def __init__(self, params):
        super().__init__(get_network(params))
