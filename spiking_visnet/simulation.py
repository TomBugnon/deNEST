#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

from .session import Session
from .utils.structures import traverse


class Simulation:
    """Represents a simulation.

    Handles building the network, running it with a series of sessions, and
    saving output.

    Args:
        params (dict-like): The VisNet parameters specifying the simulation.
    """

    def __init__(self, params):
        """Initialize."""
        self.order = params['sessions_order']
        self.sessions = {
            name: Session(session_params)
            for name, session_params in traverse(params)
        }

    def run(self, params, network):
        """Run each of the sessions in order."""
        for name in self.order:
            print(f'Running session `{name}`...')
            self.sessions[name].run(network)

    def save_sessions(self, save_dir):

        for session_name, session in self.sessions.items():
            session.save_stim(save_dir, session_name)
