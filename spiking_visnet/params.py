#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# params.py

"""Provides the ``Params`` class."""

import collections
import functools
import operator


class Params(collections.UserDict):
    """Represent simulation parameters.

    Same as a dictionary, but:
        - Allows getting and setting nested values with a tuple of keys
        - Allows accessing keys that don't exist; the default value is another
            ``Params`` instance
    """

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return functools.reduce(operator.getitem, key, self)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self[key[:-1]][key[-1]] = value
        else:
            super().__setitem__(key, value)

    def __missing__(self, key):
        self[key] = Params()
        return self[key]
