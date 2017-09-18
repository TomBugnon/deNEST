#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

import os
from os.path import join
from shutil import rmtree

from .nestify.build import Network
from .save import save_as_yaml
from .session import Session
from .user_config import NEST_SEED, OUTPUT_DIR, PYTHON_SEED


class Simulation:
    """Represents a simulation.

    Handles building the network, running it with a series of sessions, and
    saving output.

    Args:
        params (dict-like): full parameter tree
    """
    def __init__(self, params):
        """Initialize simulation."""
        self.params = params
        # Get output dir and nest tmp output_dir
        self.output_dir = self.get_output_dirs()
        # set python seeds
        self.set_python_seeds()
        # Initialize kernel (should be after getting output dirs)
        print('Initialize NEST kernel...', flush=True)
        self.init_kernel()
        print('...done', flush=True)
        # Create network
        print('Create network...', flush=True)
        self.network = Network(self.params.c['network'])
        self.network.create()
        print('...done', flush=True)
        # Create sessions
        print('Create sessions...', flush=True)
        self.order = self.params.c['sessions']['order']
        self.sessions = {
            name: Session(name, session_params)
            for name, session_params in self.params.c['sessions'].named_leaves()
        }
        self.session_times = None
        print('Done...', flush=True)

    def run(self):
        """Run each of the sessions in order."""
        for name in self.order:
            print(f'Running session `{name}`...')
            self.sessions[name].run(self.network)
        # Get session times
        self.session_times = {
            session_name: session.duration
            for session_name, session in self.sessions.items()
            }

    def save(self):
        """Save simulation"""
        # Save params
        save_as_yaml(join(self.output_dir, 'params'), self.params)
        # Save network
        with_rasters = self.params.c['simulation'].get('save_nest_raster', True)
        self.network.save(self.output_dir, with_rasters = with_rasters)
        # Save sessions
        for session in self.sessions.values():
            session.save(self.output_dir)
        # Save session times
        save_as_yaml(join(self.output_dir, 'session_times'), self.session_times)
        # Delete nest temporary directory
        if self.params.c['simulation'].get('delete_tmp_dir', True):
            rmtree(self.params.c['simulation']['nest_output_dir'])

    def init_kernel(self):
        """Initialize NEST kernel."""
        import nest
        kernel_params = self.params.c['kernel']
        nest.ResetKernel()
        # Create tmp directory in advance
        tmp_dir = kernel_params['data_path']
        os.makedirs(tmp_dir, exist_ok=True)
        nest.SetKernelStatus(
            {'local_num_threads': kernel_params.get('local_num_threads', 1),
             'resolution': kernel_params.get('resolution', 1.),
             'overwrite_files': kernel_params.get('overwrite_files', True),
             'data_path': tmp_dir})
        msd = kernel_params.get('nest_seed', NEST_SEED)
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
        print(f'Set python seed: {str(python_seed)}')
        np.random.seed(python_seed)
        random.seed(python_seed)

    def get_output_dirs(self):
        """Get output_dir from params and update kernel params accordingly."""
        output_dir = self.params.c['simulation'].get('output_dir', False)
        # If not specified by USER, get default from config
        if not output_dir:
            output_dir = OUTPUT_DIR
            # Save output dir in params
            self.params.c['simulation']['output_dir'] = output_dir
        # Tell NEST kernel to save recorder files in OUTPUT_DIR/tmp
        nest_output_dir = join(output_dir, 'tmp')
        self.params.c['kernel']['data_path'] = nest_output_dir
        self.params.c['simulation']['nest_output_dir'] = nest_output_dir
        return output_dir
