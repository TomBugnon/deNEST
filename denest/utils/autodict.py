#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# autodict.py
"""Provides the ``AutoDict`` class."""

import functools
import operator
from collections import UserDict
from collections.abc import Mapping


def dictify(obj):
    """Recursively convert generic mappings to dictionaries."""
    if isinstance(obj, Mapping):
        return {key: dictify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dictify(elt) for elt in obj]
    return obj


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

    def todict(self):
        return dictify(self)
