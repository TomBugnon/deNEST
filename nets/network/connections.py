#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py
"""Connection classes."""

# pylint:disable=missing-docstring

import csv
from copy import deepcopy
from os.path import join

from ..io import save
from .nest_object import NestObject
from .utils import if_not_created

# List of Connection and ConnectionModel parameters (and their default values
# that shouldn't be considered as 'nest_parameters'
NON_NEST_CONNECTION_PARAMS = {
    'type': 'topological',  # 'Topological'
    'dump_connection': False,
    'plot_connection': False,
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


class BaseConnection(NestObject):
    """Base class for all population-to-population connections.

    A Connection consists in synapses between two populations that have a
    specific  model. Population-to-population connections are specified
    in the ``connections`` network/topology parameter. Connection models are
    specified in the ``connection_models`` network parameter.

    ``(<connection_model_name>, <source_layer_name>, <source_population_name>,
    <target_layer_name>, <target_population_name>)`` tuples fully specify each
    individual connection and should be unique. Refer to
    ``Network.build_connections`` for a description of how these arguments are
    parsed.

    Connection weights can be recorded by 'weight_recorder' devices,
    represented by ConnectionRecorder objects. Because the weight recorder
    device's GID must be specified in a synapse model's default parameters
    (using nest.SetDefaults or nest.CopyModel), the actual NEST synapse model
    used by a connection (`self.nest_synapse_model`) might be different from the
    one specified in the connection's parameters (`self.base_synapse_model`)

    Args:
        model (ConnectionModel): ConnectionModel object. Provide base
            'params' and 'nest_params' parameter dictionaries.
        source_layer, target_layer (Layer): source and target Layer object
        source_population, target_population (str | None): Name of the
            source and target population. If None, all populations are used.
            Wrapper for the `sources` and `targets` `nest.ConnectLayers`
            parameters.
    """

    # TODO: Make some methods private?
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(self, model, source_layer, source_population, target_layer,
                 target_population):
        """Initialize Connection object."""
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
        self.validate_connection()

    # Properties:
    @property
    def base_synapse_model(self):
        """Return synapse model specified in Connection's model."""
        return self._nest_synapse_model

    @property
    def nest_synapse_model(self):
        """Return synapse model used in NEST for this connection.

        May differ from self.base_synapse_model to allow recording to a
        weight_recorder.
        """
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

    # Creation and connection

    @if_not_created
    def create(self):
        """Create the connections in NEST and the connection recorders."""
        pass

    def connect_connection_recorder(self, recorder_type='weight_recorder',
                                    recorder_gid=None):
        """Create and use new synapse model connected to a ConnectionRecorder.

        The new `nest_synapse_model` will be used during creation rather than
        the `base_synapse_model` specified in the nest parameters.
        """
        import nest
        if recorder_type == 'weight_recorder':
            # If we want to record the synapse, we create a new model and change
            # its default params so that it connects to the weight recorder
            self._nest_synapse_model = self.nest_synapse_model_name()
            nest.CopyModel(
                self._base_synapse_model,
                self._nest_synapse_model,
                {
                    recorder_type: recorder_gid
                }
            )
            # Use modified synapse model for connection
            self.nest_params['synapse_model'] = self._nest_synapse_model
        else:
            raise ValueError(f"ConnectionRecorder type `{recorder_type}` not"
                             "recognized")

    def nest_synapse_model_name(self):
        return f"{self._base_synapse_model}-{self.__str__()}"

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
        pass

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

    def validate_connection(self):
        """Check the connection to avoid bad errors.

        Raise ValueError if:
            1. ``InputLayer`` layers are never targets
            2. The source population of ``InputLayer`` layers is
                ``parrot_neuron``
        """

        if type(self.target).__name__ == 'InputLayer':
            raise ValueError(
                f"Invalid target in connection {str(self)}: `InputLayer` layers"
                f" cannot be connection targets."
            )
        if (
            type(self.source).__name__ == 'InputLayer'
            and self.source_population != self.source.PARROT_MODEL
        ):
            raise ValueError(
                f"Invalid source population for connection {str(self)}: "
                f" the source population of connections must be"
                f"{self.source.PARROT_MODEL} for `InputLayer` layers"
            )


class TopoConnection(BaseConnection):
    """Represent a topological connection."""

    def __init__(self, *args):
        super().__init__(*args)

    # Creation functions not inherited from BaseConnection

    @if_not_created
    def create(self):
        """Create the connections in NEST."""
        self.source.connect(self.target, self.nest_params)
