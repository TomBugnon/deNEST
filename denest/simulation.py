#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation.py

"""Provides the ``Simulation`` class."""

import logging

from .io.save import make_output_dir, output_path, output_subdir, save_as_yaml
from .network import Network
from .parameters import ParamsTree
from .session import Session
from .utils import misc, validation

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Simulation(object):
    """Represents a simulation.

    Handles building the network, running it with a series of sessions, and
    saving output.

    Args:
        tree (ParamsTree): Full simulation parameter tree. The following
            ``ParamsTree`` subtrees are expected:

            ``simulation`` (:class:`ParamsTree`)
                Defines input and output paths, and the simulation steps
                performed. The following parameters (``params`` field) are
                recognized:
                    ``output_dir`` (str)
                      Path to the output directory. (Default: ``'output'``)
                    ``input_dir`` (str)
                      Path to the directory in which input files are searched
                      for for each session. (Default: ``'input'``)
                    ``sessions`` (list[str])
                      Order in which sessions are run. Elements of the list
                      should be the name of session models defined in the
                      ``session_models`` parameter subtree. (Default:
                      ``[]``)
            ``kernel`` (:class:`ParamsTree`)
                Used for NEST kernel initialization. Refer to
                :meth:`Simulation.init_kernel` for a description of kernel
                parameters.
            ``session_models`` (:class:`ParamsTree`)
                Parameter tree, the leaves of which define session models.
                Refer to :class:`Session` for a description of session
                parameters.
            ``network`` (:class:`ParamsTree`)
                Parameter tree defining the network in NEST. Refer to
                :class:`Network` for a full description of network
                parameters.

    Keyword Args:
        input_dir (str | None): None or the path to the input. If defined,
            overrides the ``input_dir`` simulation parameter.
        output_dir (str | None): None or the path to the output directory. If
            defined, overrides the ``output_dir`` simulation parameter.
    """

    # Validate children subtrees
    MANDATORY_CHILDREN = []
    OPTIONAL_CHILDREN = ["kernel", "simulation", "session_models", "network"]

    # Validate "simulation" params
    # TODO: Check there is no "nest_params"
    MANDATORY_SIM_PARAMS = []
    OPTIONAL_SIM_PARAMS = {
        "sessions": [],
        "input_dir": "input",
        "output_dir": "output",
    }

    def __init__(self, tree=None, input_dir=None, output_dir=None):
        """Initialize simulation.

        - Set input and output paths
        - Initialize NEST kernel
        - Initialize and build Network in NEST
        - Create sessions
        - Save simulation metadata
        """
        # Full parameter tree
        if tree is None:
            tree = ParamsTree()
        self.tree = tree.copy()

        # Validate params tree
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # Check that the full tree's data keys are empty
        validation.validate(
            "Full parameter tree",
            dict(tree.params),
            param_type="params",
            mandatory=[],
            optional={},
        )
        validation.validate(
            "Full parameter tree",
            dict(tree.nest_params),
            param_type="nest_params",
            mandatory=[],
            optional={},
        )
        # Check that the full parameter tree has the correct children
        validation.validate_children(
            self.tree,
            mandatory_children=self.MANDATORY_CHILDREN,
            optional_children=self.OPTIONAL_CHILDREN,
        )
        # Validate "simulation" subtree
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        simulation_tree = self.tree.children["simulation"]
        # No nest_params in `simulation` subtree
        validation.validate(
            "simulation",
            dict(simulation_tree.nest_params),
            mandatory=[],
            optional={},
            param_type="nest_params",
        )
        # No children in `simulation` subtree
        validation.validate_children(
            simulation_tree, mandatory_children=[], optional_children=[]
        )
        # Validate `params` and save in `simulation` subtree
        self.sim_params = validation.validate(
            "simulation",
            dict(simulation_tree.params),
            mandatory=self.MANDATORY_SIM_PARAMS,
            optional=self.OPTIONAL_SIM_PARAMS,
        )

        # Incorporate `input_dir` and `output_dir` kwargs
        if output_dir is not None:
            self.sim_params["output_dir"] = output_dir
            self.tree.children['simulation'].params['output_dir'] \
                = str(output_dir)
        self.output_dir = self.sim_params["output_dir"]
        # Get input dir
        if input_dir is not None:
            self.sim_params["input_dir"] = input_dir
            self.tree.children['simulation'].params['input_dir'] \
                = str(input_dir)
        self.input_dir = self.sim_params["input_dir"]

        # Initialize kernel (should be after getting output dirs)
        self.init_kernel(self.tree.children['kernel'])

        # Create session models
        self.session_models = None
        self.build_session_models(self.tree.children['session_models'])

        # Create sessions
        self.sessions = None
        self.session_times = None
        self.build_sessions(self.sim_params['sessions'])

        # Create network
        self.network = None
        self.create_network(self.tree.children["network"])

        # Save simulation metadata
        self.save_metadata(clear_output_dir=True)

    def _update_tree_child(self, child_name, tree):
        """Add a child to ``self.tree``"""
        # Convert to ParamsTree and specify parent tree to preserve inheritance
        if not isinstance(tree, ParamsTree):
            child_tree = ParamsTree(tree, parent=self.tree, name=child_name)
        else:
            child_tree = ParamsTree(
                tree.asdict(),
                parent=self.tree,
                name=child_name
            )
        # Add as child
        self.tree.children[child_name] = child_tree

    def create_network(self, network_tree):
        """Build and create the network.

        Adds ``network_tree`` as ``'network'`` child of ``self.tree``
        """
        # Add to self.tree
        self._update_tree_child('network', network_tree)

        log.info("Building network.")
        self.network = Network(network_tree)
        log.info("Creating network.")
        self.network.create()
        log.info("Finished creating network")

    def save_metadata(self, clear_output_dir=False):
        """Save simulation metadata.

        - Save parameters
        - Save deNEST git hash
        - Save sessions metadata (:meth:`Session.save_metadata`)
        - Save session times (start and end kernel time for each session)
        - Save network metadata (:meth:`Network.save_metadata`)

        Keyword Args:
            clear_output_dir (bool): If true, delete the contents of the
                output directory.
        """
        log.info("Saving simulation metadata...")
        # Initialize output dir (create and clear)
        log.info("Creating output directory: %s", self.output_dir)
        make_output_dir(self.output_dir, clear_output_dir=clear_output_dir)
        # Save params tree
        self.tree.write(output_path(self.output_dir, "tree"))
        # Drop version information
        misc.drop_versions(self.output_dir)
        # Save sessions
        for session in self.sessions:
            session.save_metadata(self.output_dir)
        # Save session times
        save_as_yaml(output_path(self.output_dir, "session_times"), self.session_times)
        # Save network metadata
        self.network.save_metadata(self.output_dir)
        log.info("Finished saving simulation metadata")

    def run(self):
        """Run simulation.

        Run sessions in the order specified by the ``'sessions'`` simulation
        parameter.
        """
        # Get list of recorders
        log.info("Running %s sessions...", len(self.sessions))
        for session in self.sessions:
            log.info("Running session: '%s'...", session.name)
            session.run(self.network)
            log.info("Done running session '%s'", session.name)
        log.info("Finished running simulation")

    def build_sessions(self, sessions_order):
        """Build a list of sessions.

        Session params are inherited from session models.
        """
        import nest

        # Add to Simulation.tree
        self.tree.children['simulation'].params['sessions'] \
            = sessions_order

        log.info(f"Build N={len(sessions_order)} sessions")
        # Create session objects
        self.sessions = []
        session_start_time = nest.GetKernelStatus('time')
        for i, session_model in enumerate(sessions_order):
            self.sessions.append(
                Session(
                    self._make_session_name(session_model, i),
                    dict(self.session_models[session_model].params),
                    start_time=session_start_time,
                    input_dir=self.input_dir,
                )
            )
            # start of next session = end of current session
            session_start_time = self.sessions[-1].end
        self.session_times = {
            session.name: (session.start, session.end)
            for session in self.sessions
        }
        log.info(f"Sessions: %s", [session.name for session in self.sessions])

    def build_session_models(self, tree):
        """Create session models from the leaves of a tree.

        Adds ``tree`` as the ``'session_models'`` child of ``self.tree``
        """
        # Add to Simulation.tree
        self._update_tree_child('session_models', tree)
        session_model_nodes = {
            session_name: session_node
            for session_name, session_node
            in self.tree.children['session_models'].named_leaves(root=False)
        }
        # Validate session_model nodes: no nest_params
        for name, node in session_model_nodes.items():
            validation.validate(
                name,
                dict(node.nest_params),
                mandatory=[],
                optional={},
                param_type="nest_params",
            )

        self.session_models = session_model_nodes
        log.info(f"Build N={len(self.session_models.keys())} session models")

    def init_kernel(self, kernel_tree):
        """Initialize the NEST kernel.

        - Call ``nest.SetKernelStatus`` with ``nest_params``
        - Set NEST kernel ``data_path`` and seed
        - Install extension modules

        Adds ``kernel_tree`` as the ``'kernel'`` child of ``self.tree``.

        Args:
            kernel_tree (ParamsTree): Parameter tree without children. The
                following parameters (``params`` field) are recognized:
                    ``extension_modules``: (list(str))
                        List of modules to install. (Default: ``[]``)
                    ``nest_seed``: (int)
                        Used to set NEST kernel's RNG seed. (Default: ``1``)
                NEST parameters (``nest_params`` field) are passed to
                ``nest.SetKernelStatus``. The following nest parameters are
                reserved: ``[data_path, 'grng_seed', 'rng_seed']``. The NEST
                seeds should be set via the ``'nest_seed'`` kernel parameter
                parameter.
        """
        import nest

        MANDATORY_PARAMS = []
        OPTIONAL_PARAMS = {
            "extension_modules": [],
            "nest_seed": 1,
        }
        RESERVED_NEST_PARAMS = ["data_path", "msd", "grng_seed", "rng_seed"]

        # Add to Simulation.tree
        self._update_tree_child('kernel', kernel_tree)
        kernel_tree = self.tree.children["kernel"]

        # Validate "kernel" subtree
        validation.validate_children(
            kernel_tree, mandatory_children=[], optional_children=[]
        )  # No children
        # Validate params and nest_params
        params = validation.validate(
            "kernel",
            dict(kernel_tree.params),
            param_type="params",
            mandatory=MANDATORY_PARAMS,
            optional=OPTIONAL_PARAMS,
        )
        nest_params = validation.validate(
            "kernel",
            dict(kernel_tree.nest_params),
            param_type="nest_params",
            reserved=RESERVED_NEST_PARAMS,
        )

        log.info("Initializing NEST kernel and seeds...")
        log.info("  Resetting NEST kernel...")
        nest.ResetKernel()

        log.info("  Setting NEST kernel status...")
        log.info("    Calling `nest.SetKernelStatus(%s)`", nest_params)
        nest.SetKernelStatus(nest_params)
        # Set data path:
        data_path = output_subdir(self.output_dir, "raw_data", create_dir=True)
        # Set seed. Do that after after first SetKernelStatus call in case
        # total_num_virtual_procs has changed
        n_vp = nest.GetKernelStatus(["total_num_virtual_procs"])[0]
        msd = params["nest_seed"]
        kernel_params = {
            "data_path": str(data_path),
            "grng_seed": msd + n_vp,
            "rng_seeds": range(msd + n_vp + 1, msd + 2 * n_vp + 1),
        }
        log.info("    Calling `nest.SetKernelStatus(%s)", kernel_params)
        nest.SetKernelStatus(kernel_params)
        log.info("  Finished setting NEST kernel status")

        # Install extension modules
        log.info("  Installing external modules...")
        for module in params["extension_modules"]:
            self.install_module(module)
        log.info("  Finished installing external modules")

        log.info("Finished initializing kernel")

    def total_time(self):
        """Return the total duration of all sessions."""
        return self.sessions[-1].end - self.sessions[0].start

    @staticmethod
    def install_module(module_name):
        """Install module in NEST using ``nest.Install()`` and catch errors.

        Even after resetting the kernel, NEST throws a ``NESTError`` rather
        than a warning when the module is already loaded. I (Tom Bugnon)
        couldn't find a way to test whether the module is already installed
        so this function catches the error if the module is already installed
        by matching the error message.

        Args:
            module_name (str): Name of the module.
        """
        import nest

        try:
            nest.Install(module_name)
        except nest.NESTError as exception:
            if "loaded already" in str(exception):
                log.info("Module %s is already loaded", module_name)
                return
            if "could not be opened" in str(exception) and "file not found" in str(
                exception
            ):
                log.error(
                    "Module %s could not be loaded. Did you compile and install the extension module?",
                    module_name,
                )
                raise exception
            raise

    @staticmethod
    def _make_session_name(name, index):
        """Return a formatted session name comprising the session index."""
        return str(index).zfill(2) + "_" + name
