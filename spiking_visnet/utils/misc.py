#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/misc.py

"""Miscellaneous utils."""

import time


def pretty_time(start_time):
    minutes, seconds = divmod(time.time()-start_time, 60)
    hours, minutes = divmod(minutes, 60)
    return "%dh:%02dm:%02ds" % (hours, minutes, seconds)
