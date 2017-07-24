#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

from .session import Session
from .utils.structures import traverse


class Simulation:
    """Represents a series of sessions."""

    def __init__(self, params):
        """Initialize."""
        self.order = params['order']
        self.sessions = {
            name: Session(session_params)
            for name, session_params in traverse(params['sessions'])
        }

    def run(self):
        """Run each of the sessions in order."""
        for name in self.order:
            print(f'Running session `{name}`...')
            self.sessions[name].run()
