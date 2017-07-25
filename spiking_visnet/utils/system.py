#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# system.py


from os import makedirs
from os.path import isdir


def mkdir_ifnot(path):
    """Recursively create a directory at <path> if there is none."""
    if not isdir(path):
        makedirs(path, exist_ok=True)
