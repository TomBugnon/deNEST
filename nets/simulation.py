#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

import os
from shutil import rmtree

from .constants import (DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, NEST_SEED,
                        PYTHON_SEED)
from .network import Network
from .save import make_output_dir, output_path, output_subdir, save_as_yaml
from .session import Session


class Simulation:
    """Represents a simulation.

    Handles building the network, running it with a series of sessions, and
    saving output.

    Args:
        params (dict-like): full parameter tree
    """
    def __init__(self, params, input_path=None, output_dir=None):
        """Initialize simulation."""
        self.params = params
        # Get output dir and nest raw output_dir
        self.output_dir = self.get_output_dirs(output_dir)
        # Get input dir
        self.input_path = self.get_input_path(input_path)
        # set python seeds
        print('Set python seed...', flush=True)
        self.set_python_seeds()
        print('...done\n', flush=True)
        # Initialize kernel (should be after getting output dirs)
        print('Initialize NEST kernel...', flush=True)
        self.init_kernel()
        print('...done\n', flush=True)
        # Create network
        print('Create network...', flush=True)
        self.network = Network(self.params.c['network'])
        self.network.create(dry_run=self.params.c['simulation'].get('dry_run',
                                                                    False))
        print('...done\n', flush=True)
        # Create sessions
        print('Create sessions...', flush=True)
        self.order = self.params.c['sessions'].get('order', [])
        session_params = {
            session_name: session_params
            for session_name, session_params
            in self.params.c['sessions'].named_leaves()
        }
        self.sessions = [
            Session(self.make_session_name(session_name, i),
                    session_params[session_name])
            for i, session_name in enumerate(self.order)
        ]
        self.session_times = None
        print(f'-> Session order: {self.order}')
        print('Done...\n', flush=True)

    def run(self):
        """Run and save each of the sessions in order."""
        # Get list of recorders and formatting parameters
        parallel = self.params.c['simulation'].get('parallel',
                                                   True)
        n_jobs = self.params.c['simulation'].get('n_jobs',
                                                 -1)
        for session in self.sessions:
            print(f'Running session: `{session.name}`...')
            session.run(self.network)
            session.save_data(self.output_dir,
                              self.network,
                              parallel=parallel,
                              n_jobs=n_jobs)

    def dump_connections(self):
        """Dump network connections."""
        self.network.dump_connections(self.output_dir)

    def plot_connections(self):
        """Plot network connections."""
        self.network.plot_connections(self.output_dir)

    def dump_connection_numbers(self):
        """Dump connection numbers."""
        self.network.dump_connection_numbers(self.output_dir)

    def save_metadata(self):
        """Save simulation metadata before running the simulation."""
        # Initialize output dir (create and clear)
        print(f'Creating output_dir: {self.output_dir}')
        clear_output_dir = self.params.c['simulation'].get('clear_output_dir',
                                        False)
        make_output_dir(self.output_dir, clear_output_dir)
        # Save params
        save_as_yaml(output_path(self.output_dir, 'params'),
                     self.params)
        # Save network metadata
        self.network.save_metadata(self.output_dir)

    def save_data(self):
        """Save data after the simulation has been run."""
        if not self.params.c['simulation']['dry_run']:
            # Save sessions
            for session in self.sessions:
                session.save_metadata(self.output_dir)
            # Save session times
            self.session_times = {
                session.name: session.duration for session in self.sessions
            }
            save_as_yaml(output_path(self.output_dir, 'session_times'),
                         self.session_times)
            # Save network
            with_rasters = self.params.c['simulation'].get('save_nest_raster',
                                                           True)
            self.network.save_data(self.output_dir,
                                   with_rasters=with_rasters)
        # Delete nest temporary directory
        if self.params.c['simulation'].get('delete_raw_dir', True):
            rmtree(self.params.c['simulation']['nest_output_dir'])

    def init_kernel(self):
        """Initialize NEST kernel."""
        import nest
        kernel_params = self.params.c['kernel']
        nest.ResetKernel()
        # Install extension modules
        for module in kernel_params.get('extension_modules', []):
            self.install_module(module)
        # Create raw directory in advance
        raw_dir = kernel_params['data_path']
        os.makedirs(raw_dir, exist_ok=True)
        # Set kernel status
        num_threads = kernel_params.get('local_num_threads', 1)
        resolution = kernel_params.get('resolution', 1.)
        msd = kernel_params.get('nest_seed', NEST_SEED)
        print('-> NEST master seed: ', str(msd))
        print('-> data_path: ', str(raw_dir))
        print('-> local_num_threads: ', str(num_threads))
        nest.SetKernelStatus(
            {'local_num_threads': num_threads,
             'resolution': resolution,
             'overwrite_files': kernel_params.get('overwrite_files', True),
             'data_path': raw_dir})
        N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        nest.SetKernelStatus({
            'grng_seed': msd + N_vp,
            'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
            'print_time': kernel_params['print_time'],
        })

    def set_python_seeds(self):
        import numpy as np
        import random
        python_seed = self.params.c['kernel'].get('python_seed', PYTHON_SEED)
        print(f'-> Setting python seed: {str(python_seed)}')
        np.random.seed(python_seed)
        random.seed(python_seed)

    def get_output_dirs(self, output_dir=None):
        """Get output_dir from params and update kernel params accordingly."""
        if output_dir is None:
            output_dir = self.params.c['simulation'].get('output_dir', False)
        # If not specified by USER, get default from config
        if not output_dir:
            output_dir = DEFAULT_OUTPUT_DIR
            # Save output dir in params
            self.params.c['simulation']['output_dir'] = output_dir
        # Tell NEST kernel where to save the raw recorder files
        nest_output_dir = output_subdir(output_dir, 'raw')
        self.params.c['kernel']['data_path'] = nest_output_dir
        self.params.c['simulation']['nest_output_dir'] = nest_output_dir
        return output_dir

    def get_input_path(self, input_path=None):
        """Get input dir from params or defaults and cast to session params."""
        if input_path is None:
            input_path = self.params.c['simulation'].get('input_path', False)
        # If not specified by USER, get default from config
        if not input_path:
            input_path = DEFAULT_INPUT_PATH
            self.params.c['simulation']['input_path'] = input_path
        # Cast to session params as well as simulation params
        self.params.c['sessions']['input_path'] = input_path
        return input_path

    def total_time(self):
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
        except nest.NESTError as e:
            if 'loaded already' in str(e):
                print(f'Module {module_name} is already loaded.')
                return
            raise

    @staticmethod
    def make_session_name(name, index):
        """Return a formatted session name comprising the session index."""
        return str(index).zfill(2) + '_' + name
