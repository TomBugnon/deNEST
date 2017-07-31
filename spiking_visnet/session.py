#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py


"""Represent a sequence of stimuli."""


import collections

import nest


# TODO: finish
class Session(collections.UserDict):
    """Represents a sequence of stimuli."""

    def __init__(self, session_params):
        """Initialize."""
        print('create Session')
        super().__init__(session_params)

    def run(self):
        """Run."""
        print("I'm a session and I am running!")
        nest.Simulate(10.)
        return
