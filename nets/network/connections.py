#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py
"""Connection classes."""

# pylint:disable=missing-docstring

import csv
import itertools
from collections import ChainMap
from copy import deepcopy
from os.path import join

from .. import save
from .nest_object import NestObject
from .recorders import ConnectionRecorder
from .utils import if_not_created

# List of Connection and ConnectionModel parameters (and their default values
# that shouldn't be considered as 'nest_parameters'
NON_NEST_CONNECTION_PARAMS = {
    'type': 'topological',  # 'Topological'
    'dump_connection': False,
    'plot_connection': False,
    'recorders': {},
    'synapse_label': None,  # (int or None). Only for *_lbl synapse models
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
    specific synapse model.

    Population-to-population connections are described by a dictionnary of the
    following form::
    {
        `source_layer`: 'source_layer',
        `source_population`: 'source_population',
        `target_layer`: 'target_layer',
        `target_population`: 'target_population',
        `model`: 'connection_model'
        `params`: 'non-nest-parameters',
        `nest_params`: 'nest_params',
    }

    A Connection's `nest_params` and `params` are inherited and ChainMapped from
    its ConnectionModel model.
    The "non-nest" parameters (listed along with their default values in the
    `NON_NEST_CONNECTION_PARAMS`) are popped off the `nest_params` parameters
    at initialization and creation
    Connections() inherit and possibly override their parameters (using a
    ChainMap) from their respective ConnectionModel model. From their merged
    connection dictionary, certain parameters are popped off and saved as
    attributes. The remaining parameters are NEST parameters that are passed to
    the kernel during a `Connect()` or `ConnectLayers()` call.
    The parameters that shouldn't be considered as NEST parameters (and should
    therefore be removed from the parameters during initialization or creation)
    are listed in the global variable `NON_NEST_CONNECTION_PARAMS`.

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

    def __init__(self, source, target, model, connection_dict):
        """Initialize Connection object from model and overrides.

        Initialize the self.params and self.nest_params attributes, and all the
        other attributes as well :)

        Args:
            source (Layer): source Layer object
            target (Layer): target Layer object
            model (ConnectionModel): ConnectionModel object. Provide base
                'params' and 'nest_params' parameter dictionaries.
            connection_dict (dict): Dictionary defining the connection. The
                dictionary should have the form described in the class
                docstring. In particular, it may contain the following keys:
                    params (dict): "non-nest" parameter dictionary. Combined in
                        a ChainMap with `model.params`. All recognized
                        parameters are listed in global variable
                        `NON_NEST_CONNECTION_PARAMS`.
                    nest_params (dict): Parameters that may be passed to the
                        NEST kernel. Combined in a ChainMap with
                        model.nest_params. No parameter listed in global
                        variable `NON_NEST_CONNECTION_PARAMS` should be present
                        in this variable.
        """
        ##
        # Check the params and nest_params dictionaries and ChainMap them with
        # the ConnectionModel params and nest_params
        params = connection_dict.get('params', {})
        nest_params = connection_dict.get('nest_params', {})
        assert all(
            [key in NON_NEST_CONNECTION_PARAMS for key in params.keys()]
            ), (
                f'Unrecognized parameter in connection: {connection_dict}.'
                f'\nRecognized parameters: {NON_NEST_CONNECTION_PARAMS.keys()}'
            )
        assert not any(
            [key in NON_NEST_CONNECTION_PARAMS for key in nest_params.keys()]
            ), (
                f'Reserved nest parameter in connection: {connection_dict}'
                f'\"Non-nest reserved parameters: {NON_NEST_CONNECTION_PARAMS.keys()}'
            )
        self.params = dict(ChainMap(params, model.params))
        self.nest_params = dict(ChainMap(nest_params, model.nest_params))
        super().__init__(model.name, self.params)
        ##
        # Define the source/target population attributes
        self.model = model
        self.source = source
        self.source_population = connection_dict.get('source_population', None)
        self.target = target
        self.target_population = connection_dict.get('target_population', None)
        ##
        # Synapse model is retrieved from nest_params
        # The synapse model used in NEST might have a different name since we
        # need to change the default parameters of a synapse to specify the
        # weight recorder
        self._synapse_model = self.nest_params['synapse_model']
        self._nest_synapse_model = None
        ##
        # Synapse label is set in the defaults of the nest synapse model in
        # `create_nest_synapse_model` and can be used to query effectively the
        # connections of a certain preojection.
        self._synapse_label = None
        # Initialize the recorders
        self.recorders = [
            ConnectionRecorder(recorder_name, recorder_params)
            for recorder_name, recorder_params
            in self.params['recorders'].items()
        ]
        assert len(self.recorders) < 2  # Only a single recorder type so far...
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

    @property
    def __str__(self):
        return '-'.join(self.sort_key)

    @property
    def sort_key(self):
        # Mapping for sorting
        return (self.name,
                self.source.name, str(self.source_population),
                self.target.name, str(self.target_population))

    def __lt__(self, other):
        return self.sort_key < other.sort_key

    def source_gids(self):
        return self.source.gids(population=self.source_population)

    def target_gids(self):
        return self.target.gids(population=self.target_population)

    # Query stuff

    def get_recorders(self, recorder_type=None):
        for recorder in self.recorders:
            if recorder_type is None or recorder.type == recorder_type:
                yield recorder

    # Creation and connection

    @if_not_created
    def create(self):
        """Create the connections in NEST and the connection recorders.

        Should use in order the following steps:
            1- create recorders
            2- create nest_synapse_model
            3- connect
        """
        # Create recorder objects
        self.create_recorders()
        # Get the NEST synapse model (different from synapse model if we record
        # the connection
        self.create_nest_synapse_model()
        # Update the nest_parameters to get the proper connection weight, set
        # the proper nest_synapse_model, etc
        self.update_nest_params()
        # Actually create the connections in NEST
        self._connect()

    def create_recorders(self):
        """Create and connect the connection recorders."""
        conn_params = {
            "connection_name": self.__str__,
            "src_layer_name": self.source.name,
            "src_population_name": self.source_population,
            "src_gids": self.source_gids(),
            "tgt_layer_name": self.target.name,
            "tgt_population_name": self.target_population,
            "tgt_gids": self.target_gids(),
            "synapse_model": self.synapse_model,
        }
        for recorder in self.recorders:
            recorder.create(conn_params)

    def create_nest_synapse_model(self):
        """Create a new synapse model for the specific connection.

        We can change the defaults of the new `nest_synapse_model` to either:
        - send spikes to the recorder
        - set a label for that synapse for easy later query.
        """
        import nest
        # By default `nest_synapse_model` is just `synapse_model`
        self._nest_synapse_model = self.synapse_model
        if not self.recorders and self.params['synapse_label'] is None:
            return
        # If we either want to set a label, or record the synapse, we create a
        # new model and change its default params accordingly
        self._nest_synapse_model = self.nest_synapse_model_name()
        nest.CopyModel(self.synapse_model, self._nest_synapse_model)
        # Set the recorder
        if self.recorders:
            recorder = self.recorders[0]
            assert recorder.type == 'weight_recorder'
            nest.SetDefaults(self._nest_synapse_model,
                             {
                                 recorder.type: recorder.gid[0]
                             })
        # Set the synapse label
        synapse_label = self.params['synapse_label']
        if synapse_label is not None:
            assert type(self.params['synapse_label']) == int
            assert 'synapse_model' in nest.GetDefaults(self._nest_synapse_model)
            assert 'synapse_label' in nest.GetDefaults(self._nest_synapse_model), \
                    (f'\nConnection: {self.name}: \n'
                     'Attempting to set synapse label on a nest synapse model'
                     ' that does not support labels. Use a *_lbl synapse model'
                     ' (eg static_synapse_lbl)')
            nest.SetDefaults(self._nest_synapse_model,
                             {
                                 'synapse_label': self.params['synapse_label']
                             })
            self._synapse_label = synapse_label

    def nest_synapse_model_name(self):
        return f"{self.synapse_model}-{self.__str__}"

    def update_nest_params(self):
        pass

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
        self.save_recorders(output_dir)
        self.save_synapse_state(output_dir)

    def save_recorders(self, output_dir):
        """Save recorders' data."""
        for recorder in self.recorders:
            recorder.save(output_dir)

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

    def __init__(self, source, target, model, conn_dict):
        super().__init__(source, target, model, conn_dict)

    # Creation functions not inherited from BaseConnection

    def get_connections(self):
        pass

    def update_nest_params(self):
        """Update in place self.nest_params.

        - Set source and target populations,
        - Set NEST synapse model (possibly different from self.synapse_model)
        """
        # TODO: Get a view of the kernel, mask, and weights inherited from the
        # connection model
        self.set_populations_nest_params()
        self.set_synapse_model_nest_params()

    def set_synapse_model_nest_params(self):
        """Update the synapse_model given to NEST."""
        self.nest_params['synapse_model'] = self.nest_synapse_model

    def set_populations_nest_params(self):
        """Set the source and target populations in self.nest_params."""
        if self.source_population:
            self.nest_params['sources'] = {'model': self.source_population}
        if self.target_population:
            self.nest_params['targets'] = {'model': self.target_population}

    def _connect(self):
        self.source.connect(self.target, self.nest_params)
