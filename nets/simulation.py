#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

import os

from .constants import (DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, NEST_SEED,
                        PYTHON_SEED)
from .network import Network
from .io.save import make_output_dir, output_path, output_subdir, save_as_yaml
from .session import Session
from .utils import misc

# pylint:disable=missing-docstring


class Simulation:
    """Represents a simulation.

    Handles building the network, running it with a series of sessions, and
    saving output.

    Args:
        params (Params): Full simulation parameter tree. The following subtrees
            are expected:

                - ``simulation`` (dict-like). Defines input and output paths,
                    and the simulation steps performed. The following parameters
                    are expected:
                        - ``output_dir` (str): Path to the output directory.
                        - ``input_path`` (str): Path to an input file or to the
                            directory in which input files are searched for for
                            each session.
                        - ``sessions`` (list(str)): Order in which sessions are
                            run. Elements of the list should be the name of
                            session models defined in the ``session_models``
                            parameter subtree.
                - ``kernel``: Used for NEST kernel initialization. Refer to
                  ``Simulation.init_kernel`` for a description of
                  kernel parameters.
                - ``session_models``: Parameter tree, the leaves of which define
                    session models. Refer to ``Sessions`` for a description of
                    session parameters.
                - ``network``: Parameter tree defining the network in NEST.
                    Refer to `Network` for a full description of network
                    parameters.

    Kwargs:
        input_path (str | None): None or the path to the input. If defined,
            overrides the `input_path` simulation parameter
        output_dir (str | None): None or the path to the output directory. If
            defined, overrides `output_dir` simulation parameter.
     """
    def __init__(self, params, input_path=None, output_dir=None):
        """Initialize simulation.

            - Set input and output paths
            - Set python seed
            - Initialize NEST kernel
            - Initialize and build Network in NEST,
            - Create sessions
            - Save simulation metadata
        """
        self.params = params
        # Get output dir
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = params.c['simulation'].get(
                output_dir, DEFAULT_OUTPUT_DIR
            )
        # Get input dir
        if input_path is not None:
            self.input_path = input_path
        else:
            self.input_path = params.c['simulation'].get(
                input_path, DEFAULT_INPUT_PATH
            )
        # set python seeds
        print('Set python seed...', flush=True)
        self.set_python_seeds()
        print('...done\n', flush=True)
        # Initialize kernel (should be after getting output dirs)
        print('Initialize NEST kernel...', flush=True)
        self.init_kernel(self.params.c['kernel'])
        print('...done\n', flush=True)
        # Create sessions
        print('Create sessions...', flush=True)
        self.sessions_order = self.params.c['simulation'].get('sessions', [])
        session_models = {
            session_name: session_params
            for session_name, session_params
            in self.params.c['session_models'].named_leaves()
        }
        self.sessions = []
        session_start_time = 0
        for i, session_model in enumerate(self.sessions_order):
            self.sessions.append(
                Session(self.make_session_name(session_model, i),
                        session_models[session_model],
                        start_time=session_start_time,
                        input_path=self.input_path)
            )
            # start of next session = end of current session
            session_start_time = self.sessions[-1].end
        self.session_times = {
            session.name: session.duration for session in self.sessions
        }
        print(f'-> Sessions: {self.sessions_order}')
        print('Done...\n', flush=True)
        # Create network
        print('Create network...', flush=True)
        self.network = Network(self.params.c['network'])
        self.network.create()
        print('...done\n', flush=True)
        # Save simulation metadata
        print('Saving simulation metadata...', flush=True)
        self.save_metadata()
        print('...done\n', flush=True)

    def save_metadata(self):
        """Save simulation metadata.

            - Save parameters
            - Save NETS git hash
            - Save sessions metadata (`Session.save_metadata`)
            - Save session times (start and end kernel time for each session)
            - Save network metadata (`Network.save_metadata`)
        """
        # Initialize output dir (create and clear)
        print(f'Creating output_dir: {self.output_dir}')
        make_output_dir(self.output_dir,
                        clear_output_dir=True)
        # Save params
        save_as_yaml(output_path(self.output_dir, 'params'),
                     self.params)
        # Drop git hash
        misc.drop_git_hash(self.output_dir)
        # Save sessions
        for session in self.sessions:
            session.save_metadata(self.output_dir)
        # Save session times
        save_as_yaml(output_path(self.output_dir, 'session_times'),
                     self.session_times)
        # Save network metadata
        self.network.save_metadata(self.output_dir)

    def run(self):
        """Run simulation.

            - Run sessions in the order specified by the `sessions` simulation
                parameter
        """
        # Get list of recorders
        for session in self.sessions:
            print(f'Running session: `{session.name}`...\n')
            session.run(self.network)
            print(f'Done running session `{session.name}`\n\n')

    def init_kernel(self, kernel_params):
        # TODO Document
        """Initialize NEST kernel.

        Args:
            kernel_params (Params): Kernel parameters.
        """
        import nest
        nest.ResetKernel()
        # Install extension modules
        print('->Installing external modules...', end=' ')
        for module in kernel_params.get('extension_modules', []):
            self.install_module(module)
        print('done')
        # Create raw directory in advance
        print('->Creating raw data directory...', end=' ')
        data_path = output_subdir(self.output_dir, 'raw_data', create_dir=True)
        print('done')
        # Set kernel status
        print('->Setting kernel status...', end=' ')
        num_threads = kernel_params.get('local_num_threads', 1)
        resolution = kernel_params.get('resolution', 1.)
        msd = kernel_params.get('nest_seed', NEST_SEED)
        print('-> NEST master seed: ', str(msd))
        print('-> data_path: ', str(data_path))
        print('-> local_num_threads: ', str(num_threads))
        nest.SetKernelStatus(
            {'local_num_threads': num_threads,
             'resolution': resolution,
             'overwrite_files': kernel_params.get('overwrite_files', True),
             'data_path': data_path})
        n_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        nest.SetKernelStatus({
            'grng_seed': msd + n_vp,
            'rng_seeds': range(msd + n_vp + 1, msd + 2 * n_vp + 1),
            'print_time': kernel_params['print_time'],
        })
        print('done')

    def set_python_seeds(self):
        import numpy as np
        import random
        python_seed = self.params.c['kernel'].get('python_seed', PYTHON_SEED)
        print(f'-> Setting python seed: {str(python_seed)}')
        np.random.seed(python_seed)
        random.seed(python_seed)

    @staticmethod
    def total_time():
        import nest
        return nest.GetKernelStatus('time')

    @staticmethod
    def install_module(module_name):
        """Install module in NEST using nest.Install() and catch errors.

        Even after resetting the kernel, NEST throws a NESTError (rather than a)
        warning when the module is already loaded. I couldn't find a way to test
        whether the module is already installed so this function catches the
        error if the module is already installed by matching the error message.
        """
        import nest
        try:
            nest.Install(module_name)
        except nest.NESTError as exception:
            if 'loaded already' in str(exception):
                print(f'\nModule {module_name} is already loaded.')
                return
            if (
                'could not be opened' in str(exception)
                and 'file not found' in str(exception)
            ):
                print(f'\nModule {module_name} could not be loaded. Did you'
                      f' compile and install the extension module?')
                raise exception
            raise

    @staticmethod
    def make_session_name(name, index):
        """Return a formatted session name comprising the session index."""
        return str(index).zfill(2) + '_' + name
