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


class HangingSimulationError(Exception):
    def __init__(self, hanging_time):
        self.hanging_time = hanging_time
        self.message = f"Simulation froze at {hanging_time}ms"
        super().__init__(self.message)

def nest_run(sim_time):
    import nest, time
    t = time.time()
    nest.Run(sim_time)
    return time.time() - t

def simulate_in_steps(simulation_time):
    """Simulate in steps and catch hanging simulations."""
    import nest
    step = 10 # (virtual time)
    ministep = nest.GetKernelStatus('resolution')
    N_break = 3
    split_t = 1 # (real time) (s) Run next chunk in mini steps if previous took longer
    soft_break_t = 0.3 # (real time) (s) break if `N_break` consecutive mini-steps takes longer
    hard_break_t = 1 # (real time) (s) break if any mini-step takes longer
    total_sim_time = 0
    kernel_time = nest.GetKernelStatus('time')
    run_ministeps = False
    with nest.RunManager():
        while total_sim_time < simulation_time:
            next_step_t = min(step, simulation_time - total_sim_time)
            if not run_ministeps:
                step_real_t = nest_run(next_step_t)
                total_sim_time += next_step_t
                print(step_real_t)
                if step_real_t > split_t:
                    run_ministeps = True
                    print(f"Step took {step_real_t}, split next")
            else:
                # Run next {step}ms in ministeps
                # Break if 3 consecutive ministeps last too long
                # continue running in ministeps if any ministep lasts too long
                consecutive_count = 0
                total_count = 0
                total_ministep_t = 0
                print(f"Running next {step}ms in ministeps")
                for _ in range(int(step/ministep)):
                    if simulation_time == total_sim_time:
                        break
                    next_ministep_t = min(ministep, simulation_time - total_sim_time)
                    t = nest_run(next_ministep_t)
                    print(t)
                    total_sim_time += next_ministep_t
                    total_ministep_t += t
                    if t > soft_break_t:
                        print(f'ministep took {t}')
                        total_count += 1
                        consecutive_count += 1
                    else:
                        consecutive_count = 0
                    if t > hard_break_t or consecutive_count >= N_break:
                        raise HangingSimulationError(nest.GetKernelStatus('time'))
                if total_count > 0 or total_ministep_t > split_t:
                    print(f'Rerun ministeps: {consecutive_count}, {total_ministep_t}')
                    run_ministeps = True
                else:
                    run_ministeps = False
    assert nest.GetKernelStatus('time') - kernel_time == simulation_time


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
                - ``shift_origin`` (bool): If True, the ``origin`` flag of the
                    stimulation devices of all the network's ``InputLayer``
                    layers is set to the start of the session during
                    initialization. Useful to repeat sessions when the
                    stimulators are eg spike generators.
                - ``unit_changes`` (list): List describing the changes applied
                    to certain units before the start of the session.
                    Passed to ``Network.set_state``.
                - ``synapse_changes`` (list): List describing the changes
                    applied to certain synapses before the start of the session.
                    Passed to ``Network.set_state``. Refer to that
                    method for a description of how ``synapse_changes`` is
                    formatted and interpreted. No changes happen if empty.
                    (default [])

    Keyword Args:
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
        "shift_origin": False,
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

            1. Reset Network (`reset_network` parameter)
            2. Inactivate recorders (`record` parameter)
            3. Shift stimulator devices 'origin' flag to start of session
                (`shift_origin` parameter)
            4. Change network's dynamic variables by calling the
                `Network.set_state` function (`unit_changes` and
                `synapse_changes` parameter)

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
        # Reset network
        if self.params["reset_network"]:
            self.reset()

        # Inactivate all the recorders and projection_recorders for
        # `self._simulation_time`
        if not self.params["record"]:
            self.inactivate_recorders(network)

        # Inactivate all the recorders and projection_recorders for
        # `self._simulation_time`
        if self.params["shift_origin"]:
            self.shift_stimulator_origin(network)

        # Change dynamic variables
        network.set_state(
            unit_changes=self.params["unit_changes"],
            synapse_changes=self.params['synapse_changes'],
            input_dir=self.input_dir,
        )

    @staticmethod
    def reset():
        """Call `nest.ResetNetwork()`"""
        import nest
        nest.ResetNetwork()

    def shift_stimulator_origin(self, network):
        """Set 'origin' of all InputLayer devices to start of the session.

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
        import nest
        log.info(
            f"Setting `origin` flag to `{self.start}` for all stimulation "
            f"devices in ``InputLayers`` for session `{self.name}`"
        )
        stim_gids = []
        for inputlayer in network._get_layers(layer_type='InputLayer'):
            stim_gids += inputlayer.gids(
                population=inputlayer.stimulator_model
            )
        # Check all simulation devices have the same origin value
        assert len(set(nest.GetStatus(stim_gids, 'origin'))) == 1
        # Check all simulation devices have the same origin value
        assert len(set(nest.GetStatus(stim_gids, 'origin'))) == 1
        # Set origin
        nest.SetStatus(stim_gids, {'origin': self.start})

    def inactivate_recorders(self, network):
        """Set 'start' of all (projection_)recorders at the end of session.

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
        """Initialize and run session.

        Session initialization consists in the following steps:
            1. Reset Network (`reset_network` parameter)
            2. Inactivate recorders (`record` parameter)
            3. Shift stimulator devices 'origin' flag to start of session
                (`shift_origin` parameter)
            4. Change network's dynamic variables by calling the
                `Network.set_state` function (`unit_changes` and
                `synapse_changes` parameter)

        After initialization, the simulation is run for `self.simulation_time`
        msec

        Args:
            self (Session): ``Session`` object
            network (Network): ``Network`` object.
        """
        import nest

        assert self.start == int(nest.GetKernelStatus("time"))
        log.info("Initializing session...")
        self.initialize(network)
        log.info("Finished initializing session\n")
        log.info("Running session '%s' for %s ms", self.name, self.simulation_time)
        start_real_time = time.time()
        # nest.Simulate(self.simulation_time)
        simulate_in_steps(self.simulation_time)
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
