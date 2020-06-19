#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# session.py

"""Represent a sequence of stimuli."""

import logging
import time
from pprint import pformat

from .base_object import ParamObject
from .utils.misc import pretty_time
from .utils.validation import ParameterError

# pylint:disable=missing-docstring


log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Session(ParamObject):
    """Represents a sequence of stimuli.

    Args:
        name (str): Name of the session
        params (dict-like): Dictionary specifying session parameters. The
            following keys are recognized:
                - ``simulation_time`` (float): Duration of the session in ms.
                    (mandatory)
                - ``reset_network`` (bool): If true, ``nest.ResetNetwork()`` is
                    called during session initialization (default ``False``)
                - ``record`` (bool): If false, the ``start_time`` field of
                    recorder nodes in NEST is set to the end time of the
                    session, so that no data is recorded during the session
                    (default ``True``)
                - ``unit_changes`` (list): List describing the changes applied
                    to certain units before the start of the session.
                    Passed to ``Network.change_unit_states``. Refer to that
                    method for a description of how ``unit_changes`` is
                    formatted and interpreted. No changes happen if empty.
                    (default [])
                - ``synapse_changes`` (list): List describing the changes
                    applied to certain synapses before the start of the session.
                    Passed to ``Network.change_synapse_changes``. Refer to that
                    method for a description of how ``synapse_changes`` is
                    formatted and interpreted. No changes happen if empty.
                    (default [])

    Kwargs:
        start_time (float): Time of kernel in ms when the session starts
            running.
        input_dir (str): Path to the directory in which input files are searched
            for for each session.
    """

    # Validation of `params`
    RESERVED_PARAMS = None
    MANDATORY_PARAMS = ["simulation_time"]
    OPTIONAL_PARAMS = {
        "reset_network": False,
        "record": True,
        "unit_changes": [],
        "synapse_changes": [],
    }

    def __init__(self, name, params, start_time=None, input_dir=None):
        log.info('Creating session "%s"', name)
        # Sets self.name / self.params  and validates params
        super().__init__(name, params)
        self.input_dir = input_dir
        # Initialize the session start and end times
        if start_time is None:
            import nest

            start_time = nest.GetKernelStatus("time")
        self._start = start_time
        self._simulation_time = int(self.params["simulation_time"])
        if not self._simulation_time >= 0:
            raise ParameterError(
                f"Parameter `simulation_time` of session {name} should be" f" positive."
            )
        self._end = self._start + self._simulation_time

    @property
    def end(self):
        """Return kernel time at session's end."""
        return self._end

    @property
    def start(self):
        """Return kernel time at session's start."""
        return self._start

    def __repr__(self):
        return "{classname}({name}, {params})".format(
            classname=type(self).__name__, name=self.name, params=pformat(self.params)
        )

    def initialize(self, network):
        """Initialize session.

            1. Reset Network
            2. Change network's dynamic variables.
            3. (possibly) inactivate recorders

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
        # Reset network
        if self.params["reset_network"]:
            network.reset()

        # Change dynamic variables
        # TODO
        # network.change_synapse_states(self.params["synapse_changes"])
        # network.change_unit_states(self.params["unit_changes"])

        # Inactivate all the recorders and connection_recorders for
        # `self._simulation_time`
        if not self.params["record"]:
            self.inactivate_recorders(network)

    def inactivate_recorders(self, network):
        """Set 'start' of all (connection_)recorders at the end of session.

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
        # TODO: We need to do this differently if we start playing with the
        # `origin` flag of recorders, eg to repeat experiments. Hence the
        # safeguard:
        import nest

        for recorder in network.get_recorders():
            assert nest.GetStatus(recorder.gid, "origin")[0] == 0.0
        log.debug("Inactivating all recorders for session %s", self.name)
        # Set start time in the future
        network._recorder_call(
            "set_status",
            {"start": nest.GetKernelStatus("time") + self._simulation_time},
        )

    def run(self, network):
        """Initialize and run session."""
        import nest

        assert self.start == int(nest.GetKernelStatus("time"))
        log.info("Initializing session...")
        self.initialize(network)
        log.info("Finished initializing session\n")
        log.info("Running session '%s' for %s ms", self.name, self.simulation_time)
        start_real_time = time.time()
        nest.Simulate(self.simulation_time)
        log.info("Finished running session")
        log.info(
            "Session '%s' virtual running time: %s ms", self.name, self.simulation_time
        )
        log.info(
            "Session '%s' real running time: %s",
            self.name,
            pretty_time(start_real_time),
        )
        assert self.end == int(nest.GetKernelStatus("time"))

    # TODO
    def save_metadata(self, output_dir):
        """Save session metadata."""
        pass

    @property
    def simulation_time(self):
        return self._simulation_time
