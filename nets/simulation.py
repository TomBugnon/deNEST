#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

from .network import Network
from .io.save import make_output_dir, output_path, output_subdir, save_as_yaml
from .session import Session
from .utils import misc, validation

# pylint:disable=missing-docstring


class Simulation(object):
    """Represents a simulation.

    Handles building the network, running it with a series of sessions, and
    saving output.

    Args:
        tree (Tree): Full simulation parameter tree. The following
            ``Tree`` subtrees are expected:

                - ``simulation`` (``Tree``). Defines input and output paths,
                    and the simulation steps performed. The following parameters
                    (`params` field) are recognized:
                        - ``output_dir` (str): Path to the output directory
                            (default 'output').
                        - ``input_path`` (str): Path to an input file or to the
                          directory in which input files are searched for for
                          each session. If ``input_path`` points towards a
                          loadable numpy input array, it will be used for
                          setting the `InputLayer` layers' input. Otherwise,
                          ``input_path`` is interpreted as a directory in which
                          input array files are searched. (default 'input')
                        - ``sessions`` (list(str)): Order in which sessions are
                            run. Elements of the list should be the name of
                            session models defined in the ``session_models``
                            parameter subtree (default [])
                - ``kernel`` (``Tree``): Used for NEST kernel initialization.
                    Refer to ``Simulation.init_kernel`` for a description of
                    kernel parameters.
                - ``session_models`` (``Tree``): Parameter tree, the leaves of
                    which define session models. Refer to ``Sessions`` for a
                    description of session parameters.
                - ``network`` (``Tree``): Parameter tree defining the network
                    in NEST. Refer to `Network` for a full description of
                    network parameters.

    Kwargs:
        input_path (str | None): None or the path to the input. If defined,
            overrides the `input_path` simulation parameter
        output_dir (str | None): None or the path to the output directory. If
            defined, overrides `output_dir` simulation parameter.
    """

    # Validate children subtrees
    MANDATORY_CHILDREN = ['kernel', 'simulation', 'session_models', 'network']

    # Validate "simulation" params
    # TODO: Check there is no "nest_params"
    MANDATORY_SIM_PARAMS = []
    OPTIONAL_SIM_PARAMS = {
        'sessions': [],
        'input_path': 'input',
        'output_dir': 'output',
    }

    def __init__(self, tree, input_path=None, output_dir=None):
        """Initialize simulation.

            - Set input and output paths
            - Initialize NEST kernel and set python seed
            - Initialize and build Network in NEST,
            - Create sessions
            - Save simulation metadata
        """
        # Full parameter tree
        self.tree = tree

        # Validate params tree
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # Check that the full tree's data keys are empty
        validation.validate(
            "Full parameter tree", dict(tree.params), param_type='params',
            mandatory=[], optional={})
        validation.validate(
            "Full parameter tree", dict(tree.nest_params),
            param_type='nest_params', mandatory=[], optional={})
        # Check that the full parameter tree has the correct children
        validation.validate_children(
            'Full parameter tree', list(tree.children.keys()),
            mandatory_children=self.MANDATORY_CHILDREN
        )
        # Validate "simulation" subtree
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        simulation_tree = self.tree.children['simulation']
        # No nest_params in `simulation` subtree
        validation.validate(
            "simulation", dict(simulation_tree.nest_params),
            mandatory=[], optional={}, param_type='nest_params'
        )
        # No children in `simulation` subtree
        validation.validate_children(
            'simulation', list(simulation_tree.children.keys()),
            mandatory_children={}
        )
        # Validate `params` and save in `simulation` subtree
        self.sim_params = validation.validate(
            "simulation", dict(simulation_tree.params),
            mandatory=self.MANDATORY_SIM_PARAMS,
            optional=self.OPTIONAL_SIM_PARAMS
        )

        # Incorporate `input_path` and `output_dir` kwargs
        if output_dir is not None:
            self.sim_params['output_dir'] = output_dir
        self.output_dir = self.sim_params['output_dir']
        # Get input dir
        if input_path is not None:
            self.sim_params['input_path'] = input_path
        self.input_path = self.sim_params['input_path']

        # Initialize kernel (should be after getting output dirs)
        print('Initialize NEST kernel and seeds...', flush=True)
        kernel_tree = self.tree.children['kernel']
        # Validate "kernel" subtree
        # No children in `kernel` subtree
        validation.validate_children(
            'kernel', list(kernel_tree.children.keys()),
            mandatory_children={}
        )
        self.init_kernel(
            dict(kernel_tree.params),
            dict(kernel_tree.nest_params)
        )
        print('...done\n', flush=True)

        # Create sessions
        print('Create sessions...', flush=True)
        self.sessions_order = self.sim_params['sessions']
        # Get session model params
        session_model_nodes = {
            session_name: session_node
            for session_name, session_node
            in self.tree.children['session_models'].named_leaves()
        }
        # Validate session_model nodes: no nest_params
        for name, node in session_model_nodes.items():
            validation.validate(
                name, dict(node.nest_params),
                mandatory=[], optional={}, param_type='nest_params'
            )
        # Create session objects
        self.sessions = []
        session_start_time = 0
        for i, session_model in enumerate(self.sessions_order):
            self.sessions.append(
                Session(self.make_session_name(session_model, i),
                        dict(session_model_nodes[session_model].params),
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
        self.network = Network(self.tree.children['network'])
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
        # Save params tree
        save_as_yaml(output_path(self.output_dir, 'tree'),
                     self.tree)
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

    def init_kernel(self, params, nest_params):
        """Initialize NEST kernel and set Python seed

            - Call ``nest.SetKernelStatus`` with ``nest_params``
            - Set NEST kernel ``data_path`` and seed
            - Set Python rng seed for ``numpy`` and ``random`` packages
            - Install extension modules

        Args:
            params (dict-like): Kernel parameters. The following parameters are
                recognized:
                    extension_modules (list(str)): List of modules to install.
                        (default [])
                    nest_seed (int): Used to set NEST kernel's rng seed (default
                        1)
                    python_seed (int): Seed in Python ``numpy`` and ``random``
                        packages. (default 1)
            nest_params (dict-like): Kernel "NEST" parameters, passed to
                ``nest.SetKernelStatus``. The following parameters are reserved:
                ``[data_path, 'grng_seed', 'rng_seed']``. The NEST seeds should
                be set via the ``nest_seed`` kernel parameter parameter.
        """

        MANDATORY_PARAMS = []
        OPTIONAL_PARAMS = {
            'extension_modules': [],
            'nest_seed': 1,
            'python_seed': 1
        }
        RESERVED_NEST_PARAMS = ['data_path', 'msd', 'grng_seed', 'rng_seed']

        # Validate params and nest_params
        params = validation.validate(
            "kernel", params, param_type='params', mandatory=MANDATORY_PARAMS,
            optional=OPTIONAL_PARAMS
        )
        nest_params = validation.validate(
            "kernel", nest_params, param_type='nest_params',
            reserved=RESERVED_NEST_PARAMS
        )

        import nest
        nest.ResetKernel()

        print(f'-> Setting NEST kernel status')
        print(f'-->Call `nest.SetKernelStatus({nest_params})`', end=' ')
        nest.SetKernelStatus(nest_params)
        # Set data path:
        data_path = output_subdir(self.output_dir, 'raw_data', create_dir=True)
        # Set seed. Do that after after first SetKernelStatus call in case
        # total_num_virtual_procs has changed
        n_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        msd = params['nest_seed']
        kernel_params = {
            'data_path': data_path,
            'grng_seed': msd + n_vp,
            'rng_seeds': range(msd + n_vp + 1, msd + 2 * n_vp + 1),
        }
        print(f'-->Call `nest.SetKernelStatus({kernel_params})`', end=' ')
        nest.SetKernelStatus(nest_params)
        print('done')

        # Install extension modules
        print('->Installing external modules...', end=' ')
        for module in params['extension_modules']:
            self.install_module(module)
        print('done')

        # Set python seed
        import numpy as np
        import random
        python_seed = params['python_seed']
        print(f'-> Setting Python seed: {python_seed}')
        np.random.seed(python_seed)
        random.seed(python_seed)

    @staticmethod
    def total_time():
        """Return the NEST kernel time."""
        import nest
        return nest.GetKernelStatus('time')

    @staticmethod
    def install_module(module_name):
        """Install module in NEST using nest.Install() and catch errors.

        Even after resetting the kernel, NEST throws a NESTError (rather than a)
        warning when the module is already loaded. I (Tom) couldn't find a way
        to test whether the module is already installed so this function catches
        the error if the module is already installed by matching the error
        message.

        Args:
            module_name (str): Name of the module.
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
