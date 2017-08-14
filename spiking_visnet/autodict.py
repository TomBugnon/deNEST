#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# autodict.py

"""Provides the ``AutoDict`` class."""

# pylint: disable=too-many-ancestors

from collections import UserDict
import functools
import operator


class AutoDict(UserDict):
    """A dictionary supporting deep access and modification with tuples.

    Intermediate dictionaries are created if necessary.
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
        value = self[key] = type(self)()
        return value
