#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Represent a series of sessions."""


from .session import Session
from .utils.structures import traverse


class Simulation:
    """Represents running a network.

    This includes building the network from a set of parameters, running it
    with a series of sessions, and saving output.
    """

    def __init__(self, params):
        """Initialize."""
        self.order = params['sessions_order']
        self.sessions = {
            name: Session(session_params)
            for name, session_params in traverse(params)
        }

    def run(self, params, network):
        """Run each of the sessions in order.

        If <user_input> is specified, each session will use that stimulus and
        not their 'session_stims' parameter.

        """
        for name in self.order:
            print(f'Running session `{name}`...')
            self.sessions[name].run(params, network)
