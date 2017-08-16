#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

"""Provides the ``AutoDict`` class."""

import functools
import operator
from collections import UserDict
from os.path import abspath as _abspath
from os.path import dirname as _dirname

from .save import load_yaml
from .utils.structures import chaintree


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


def load_params(path, overrides=None):
    """Load a parameter file, optionally overriding some values.

    Args:
        path (str): The filepath to load.

    Keyword Args:
        overrides (dict): A dictionary containing parameters that will take
            precedence over those in the file.

    Returns:
        AutoDict: the loaded parameters with overrides applied.

    """
    directory = _dirname(_abspath(path))
    trees = [load_yaml(directory, relative_path)
             for relative_path in load_yaml(path)]
    if overrides:
        trees.insert(0, overrides)
    return AutoDict(chaintree(trees))
