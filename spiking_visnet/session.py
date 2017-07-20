#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

import collections


# TODO: finish
class Session(collections.UserDict):
    """Represents a sequence of stimuli."""

    def __init__(self, session_params):
        print('create Session')
        super().__init__(session_params)

    def run(self):
        print("I'm a session and I am running!")
        return
