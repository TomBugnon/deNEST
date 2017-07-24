#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# system.py


from os import mkdir
from os.path import isdir


def mkdir_ifnot(path):
    if not isdir(path):
        mkdir(path)
