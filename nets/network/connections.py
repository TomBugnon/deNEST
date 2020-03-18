#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py
"""Connection classes."""

# pylint:disable=missing-docstring

import csv
from collections import ChainMap
from copy import deepcopy
from os.path import join

from .. import save
from .nest_object import NestObject
from .utils import if_not_created

# List of Connection and ConnectionModel parameters (and their default values
# that shouldn't be considered as 'nest_parameters'
NON_NEST_CONNECTION_PARAMS = {
    'type': 'topological',  # 'Topological'
    'dump_connection': False,
    'plot_connection': False,
    'save': [],
}


class ConnectionModel(NestObject):
    """Represent a NEST connection model.

    The nest parameters (`self.nest_params`) of a ConnectionModel object contain
    the base nest parameters used in Connection objects. The parameters that
    should not be considered as "nest-parameters" (listed along with their
    default values in the global variable NON_NEST_CONNECTION_PARAMS) are popped
    off the `self.nest_params` dictionary and kept in the `self.params`
    attribute.
    The population-to-population Connection objects inherit from both the params
    and the nest_params dictionaries.
    """

    def __init__(self, name, all_params):
        # Pop off the params that shouldn't be considered as NEST parameters
        nest_params = deepcopy(dict(all_params))
        params = {}
        for non_nest_param, default in NON_NEST_CONNECTION_PARAMS.items():
            params[non_nest_param] = nest_params.pop(non_nest_param, default)
        # We now save the params and nest_params dictionaries as attributes
        super().__init__(name, params)
        self.nest_params = nest_params
        # Check that the connection types are recognized and nothing is missing.
        assert self.type in ['topological']

    @property
    def type(self):
        return self.params['type']

    @property
    def source_dir(self):
        return self.params['source_dir']


class BaseConnection(NestObject):
    """Base class for all population-to-population connections.

    A Connection consists in synapses between two populations that have a
    specific synapse model. Population-to-population connections are specified
    in the ``connections`` network/topology parameter.

    ``(<connection_model_name>, <source_layer_name>, <source_population_name>,
    <target_layer_name>, <target_population_name>)`` tuples fully specify each
    individual connection and should be unique. Refer to
    ``Network.build_connections`` for a description of how these arguments are
    parsed.

    Connection weights can be recorded by 'weight_recorder' devices. Because
    the weight recorder device's GID must be specified in a synapse model's
    default parameters (using nest.SetDefaults or nest.CopyModel), the actual
    NEST synapse model used when connecting might differ from the one specified
    in the network's synapse models.

    The workflow for creating connections and their respective recorders is as
    follows:
        1- Initialize the Connection object and possibly its Recorder object
        2- Create the Recorder object.
        3- Get the GID of the Recorder object and create a new NEST synapse
            model that will send the spikes to the recorder. The name of the
            synapse model is saved in self.nest_synapse_model.
        4- Create the connection with the self.nest_synapse_model model.
    self.nest_synapse_model is only used to communicate with the kernel. The
    Connection is still denoted by its source and target population and its
    "base" synapse model (saved in self.synapse_model)
    """

    # TODO: Make some methods private?
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(self, model, source_layer, source_population, target_layer,
                 target_population):
        """Initialize Connection object from model and overrides.

        Initialize the self.params and self.nest_params attributes, and all the
        other attributes as well :)

        Args:
            model (ConnectionModel): ConnectionModel object. Provide base
                'params' and 'nest_params' parameter dictionaries.
            source_layer, target_layer (Layer): source and target Layer object
            source_population, target_population (str | None): Name of the
                source and target population. If None, all populations are used.
                Wrapper for the `sources` and `targets` `nest.ConnectLayers`
                parameters.
        """
        ##
        # Inherit params and nest_params from connection model
        self.params = dict(model.params)
        self.nest_params = dict(model.nest_params)
        super().__init__(model.name, self.params)
        ##
        # Define the source and targets
        self.model = model
        self.source = source_layer
        self.target = target_layer
        self.source_population = source_population
        if self.source_population:
            self.nest_params['sources'] = {'model': self.source_population}
        self.target_population = target_population
        if self.target_population:
            self.nest_params['targets'] = {'model': self.target_population}
        #
        # Base synapse model is retrieved from nest_params. If a weight_recorder
        # is created for this connection, a different synapse model will be used
        self._base_synapse_model = self.nest_params['synapse_model']
        self._nest_synapse_model = self.nest_params['synapse_model']
        self.check()

    # Properties:
    @property
    def synapse_model(self):
        return self._synapse_model

    @property
    def nest_synapse_model(self):
        return self._nest_synapse_model

    @property
    def dump_connection(self):
        return self.params['dump_connection']

    @property
    def plot_connection(self):
        return self.params['plot_connection']

    def __str__(self):
        return '-'.join([
            self.name,
            self.source.name, str(self.source_population),
            self.target.name, str(self.target_population),
        ])

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def source_gids(self):
        return self.source.gids(population=self.source_population)

    def target_gids(self):
        return self.target.gids(population=self.target_population)

    # Creation and connection

    @if_not_created
    def create(self):
        """Create the connections in NEST and the connection recorders."""
        pass

    def create_nest_synapse_model(self, recorder_type=None, recorder_gid=None):
        """Create and use new synapse model connected to a ConnectionRecorder.

        The new `nest_synapse_model` will be used during creation rather than
        the `base_synapse_model` specified in the nest parameters.
        """
        import nest
        # If we want to record the synapse, we create a new model and change its
        # default params so that it connects to the weight recorder
        self._nest_synapse_model = self.nest_synapse_model_name()
        nest.CopyModel(
            self._base_synapse_model,
            self._nest_synapse_model,
            {
                recorder_type: recorder_gid
            }
        )
        # Connect using new synapse model
        self.nest_params['synapse_model'] = self._nest_synapse_model

    def nest_synapse_model_name(self):
        return f"{self.synapse_model}-{self.__str__}"

    # Connection dumping  to file

    def dump(self, output_dir):
        # TODO: Query using synapse labels to identify connections with same
        # source pop, target pop and synapse model
        import nest
        if self.dump_connection:
            conns = nest.GetConnections(
                source=self.source.gids(population=self.source_population),
                target=self.target.gids(population=self.target_population),
                synapse_model=self.nest_synapse_model)
            # We save: source_gid, target_gid, synapse_model, weight, delay
            with open(join(save.output_subdir(output_dir, 'dump'),
                           self.__str__), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(self.format_dump(conns))
        import warnings
        warnings.warn('Double check the synapse models vs nest_synapse_model')

    @staticmethod
    def format_dump(conns):
        import nest
        formatted = []
        for conn in conns:
            status = nest.GetStatus((conn,))[0]
            formatted.append((status['source'],
                              status['target'],
                              str(status['synapse_model']),
                              status['weight'],
                              status['delay']))
        return sorted(formatted)

    # Save and plot stuff

    def save(self, output_dir):
        self.save_synapse_state(output_dir)

    def save_synapse_state(self, output_dir):
        """Save using a GetConnections() call."""
        for field in self.params['save']:
            # TODO
            print('TODO: save connection ', field, ' in ', output_dir)

    def save_plot(self, output_dir):
        if self.plot_connection:
            import matplotlib.pyplot as plt
            fig = self.plot_conn()  # pylint: disable=unused-variable
            plt.savefig(join(save.output_subdir(output_dir, 'connections'),
                             self.__str__))
            plt.close()

    def plot_conn(self):
        """Plot the targets of a unit using nest.topology function."""
        # TODO: Get our own version so we can plot convergent connections
        import nest.topology as tp
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()  # pylint:disable=invalid-name
        tp.PlotLayer(self.target.gid, fig)
        ctr = self.source.find_center_element(population=self.source_population)
        # Plot the kernel and mask if the connection is topological
        try:
            tp.PlotKernel(ax, ctr,
                          self.nest_params['mask'],
                          kern=self.nest_params['kernel'],
                          kernel_color='green')
        except (AttributeError, KeyError, ValueError):
            # AttributeError, KeyError: if no nest_params mask or kernel
            # ValueError: if the mask or kernel cannot be plotted (custom mask)
            pass
        try:
            tp.PlotTargets(ctr,
                           self.target.gid,
                           tgt_model=self.target_population,
                           syn_type=self.synapse_model,
                           fig=fig,
                           tgt_size=40,
                           src_size=250,
                           tgt_color='red')
        except ValueError:
            print(f"Not plotting targets: the center unit {ctr[0]} has no "
                  f"target within connection {self.__str__}")
        plt.suptitle(f"Plot of targets of a single source unit.\n"
                     f"Target units' pop: {self.target.name},"
                     f"{str(self.target_population)} (targets in red),\n"
                     f"Source unit's population: {self.source.name},"
                     f"{str(self.source_population)}\n"
                     f"Connection name: {self.name},\n", fontsize=7)
        footnote = ("NB: The actual connection probability might be smaller "
                    "than it seems if there is multiple units per grid position"
                    " in the target population(s)")
        ax.annotate(footnote, xy=(1, 0), xycoords='axes fraction', fontsize=5,
                    xytext=(0, -15), textcoords='offset points',
                    ha='right', va='top')
        return fig

    def check(self):
        """Check the connection to avoid bad errors.

        Make sure that:
        - the target is not an ``InputLayer``,
        - if the source is an ``InputLayer``, the source population is a parrot
        neuron (otherwise we can't record the input layer).
        """
        assert type(self.target).__name__ != 'InputLayer'
        if (
            type(self.source).__name__ == 'InputLayer'
            and self.source_population != self.source.PARROT_MODEL
        ):
            import warnings
            warn_str = (f'\n\nCareful! The Input population for connection:'
                        f'\n{self.__str__}\n is not a parrot '
                        'neuron! This might throw a bad NEST error.\n\n\n')
            warnings.warn(warn_str)


class TopoConnection(BaseConnection):
    """Represent a topological connection."""

    def __init__(self, *args):
        super().__init__(*args)

    # Creation functions not inherited from BaseConnection

    def get_connections(self):
        pass

    @if_not_created
    def create(self):
        """Create the connections in NEST."""
        self.source.connect(self.target, self.nest_params)
