#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

from os.path import join

from .save import save_as_yaml
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
        """Save session times and stimuli."""
        # Save session times
        session_times = {
            session_name: range(session.start_time, session.end_time)
            for session_name, session in self.sessions.items()
            }
        save_as_yaml(join(save_dir, 'session_times'), session_times)
        # Save session stims
        for session_name, session in self.sessions.items():
            session.save_stim(save_dir, session_name)
