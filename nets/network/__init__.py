#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/__init__.py

"""Provide a class to construct a network."""

import logging
import itertools

from tqdm import tqdm

from ..utils import validation
from ..utils.validation import ParameterError
from .connections import ConnectionModel, TopoConnection
from .layers import InputLayer, Layer
from .models import Model, SynapseModel
from .recorders import ConnectionRecorder, PopulationRecorder
from .utils import if_not_created, log


log = logging.getLogger(__name__)


LAYER_TYPES = {
    None: Layer,
    'InputLayer': InputLayer,
}

CONNECTION_TYPES = {
    'topological': TopoConnection,
}


class Network(object):
    """Represent a full network.

    Args:
        tree (``ParamsTree``): "network" parameter tree. The following
            ``ParamsTree`` children are expected:
                - ``neuron_models`` (``ParamsTree``). Parameter tree, the leaves
                    of which define neuron models. Each leave is used to
                    initialize a ``Model`` object
                - ``synapse_models`` (``ParamsTree``). Parameter tree, the
                    leaves of which define synapse models. Each leave is used to
                    initialize a ``SynapseModel`` object
                - ``layers`` (``ParamsTree``). Parameter tree, the leaves of
                    which define layers. Each leave is used to initialize  a
                    ``Layer`` or ``InputLayer`` object depending on the value of
                    their ``type`` ``params`` parameter.
                - ``connection_models`` (``ParamsTree``). Parameter tree, the
                    leaves of which define connection models. Each leave is used
                    to initialize a ``ConnectionModel`` object.
                - ``recorder_models`` (``ParamsTree``). Parameter tree, the
                    leaves of which define recorder models. Each leave is used
                    to initialize a ``Model`` object.
                - ``topology`` (``ParamsTree``). ``ParamsTree`` object without
                    children, the ``params`` of which may contain a
                    ``connections`` key specifying all the individual
                    population-to-population connections within the network as a
                    list. ``Connection`` objects  are created from the
                    ``topology`` ``ParamsTree`` object by the
                    ``Network.build_connections`` method. Refer to this method
                    for a description of the ``topology`` parameter.
                - ``recorders`` (``ParamsTree``). ``ParamsTree`` object without
                    children, the ``params`` of which may contain a
                    ``population_recorders`` and a ``connection_recorders`` key
                    specifying all the network recorders. ``PopulationRecorder``
                    and ``ConnectionRecorder`` objects  are created from the
                    ``recorders`` ``ParamsTree`` object by the
                    ``Network.build_recorders`` method. Refer to this
                    method for a description of the ``recorders`` parameter.
    """

    MANDATORY_CHILDREN = []
    OPTIONAL_CHILDREN = [
        'neuron_models', 'synapse_models', 'layers', 'connection_models',
        'topology', 'recorder_models', 'recorders'
    ]

    def __init__(self, tree):
        """Initialize the network object without creating it in NEST."""
        self._created = False
        self._changed = False
        self.tree = tree.copy()

        # Validate tree
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # Check that the "network" tree's params and nest_params keys are empty
        validation.validate(
            "network", dict(tree.params), param_type='params', mandatory=[],
            optional={})
        validation.validate(
            "network", dict(tree.nest_params), param_type='nest_params',
            mandatory=[], optional={})
        # Check that the "network" tree has the correct children and add default
        # children
        validation.validate_children(
            self.tree, mandatory_children=self.MANDATORY_CHILDREN,
            optional_children=self.OPTIONAL_CHILDREN,
        )

        # Build network components
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.neuron_models = self.build_named_leaves_dict(
            Model, self.tree.children['neuron_models'])
        self.synapse_models = self.build_named_leaves_dict(
            SynapseModel, self.tree.children['synapse_models'])
        self.recorder_models = self.build_named_leaves_dict(
            Model, self.tree.children['recorder_models'])
        self.layers = {
            name: LAYER_TYPES[leaf.params.get('type', None)](
                name,
                dict(leaf.params),
                dict(leaf.nest_params)
            )
            for name, leaf in self.tree.children['layers'].named_leaves(
                root=False
            )
        }
        self.connection_models = self.build_named_leaves_dict(
            ConnectionModel, self.tree.children['connection_models'])
        # Connections must be built after layers and connection models
        self.connections = self.build_connections(
            self.tree.children['topology']
        )
        # Initialize population recorders and connection recorders
        self.population_recorders, self.connection_recorders = \
            self.build_recorders(
                self.tree.children['recorders']
            )

    @staticmethod
    def build_named_leaves_dict(constructor, node):
        """Construct and return as dict all leaves of a tree."""
        return {
            name: constructor(name, dict(leaf.params), dict(leaf.nest_params))
            for name, leaf in node.named_leaves(root=False)
        }

    def build_connections(self, topology_tree):
        """Return list of ``Connection`` objects from ``topology`` ParamsTree tree.

        Args:
            self (``Network``): Network object
            topology_tree (``ParamsTree``): ``ParamsTree`` object without
                children. The parameters of which may contain a ``connections``
                parameter entry (default []). THe value of the ``connections``
                parameter is a list of items describing the connections to be
                created. Each item must be a ``dict`` of the following form::
                    dict: {
                        'connection_model' : <connection_model>,
                        'source_layers': <source_layers_list>,
                        'source_population': <source_population>,
                        'target_layers': <target_layers_list>,
                        'target_population': <target_population>,
                    }
                Where:
                    - <connection_model> is the name of the connection model.
                      Connection model are specified in the
                      ``connection_models`` network parameter.
                    - <source_layers_list>, <target_layers_list> are lists of
                      source and target layer names. Connections are created for
                      all source_layer x target layer combinations.
                    - <source_population>, <target_population> are ``None`` or
                      the name of source and target populations for the created
                      connection. If ``None``, all populations in the source or
                      target layer are connected.
                The ``(<connection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>)`` tuples fully specify each individual
                connection and should be unique.

        Returns:
            list: List of ``Connection`` objects specifying all the connections
                in the network.
        """
        OPTIONAL_TOPOLOGY_PARAMS = {
            'connections': []
        }

        # Validate ``topology`` parameter
        # No children
        validation.validate_children(
            topology_tree, [], []
        )
        # No nest_params
        validation.validate(
            'topology', dict(topology_tree.nest_params),
            param_type='nest_params', mandatory=[], optional={}
        )
        # Only a 'connections' `params` entry
        connection_items = validation.validate(
            'topology', dict(topology_tree.params), param_type='params',
            mandatory=[], optional=OPTIONAL_TOPOLOGY_PARAMS
        )['connections']

        # Get all unique ``(connection_model, source_layer, source_population,
        # target_layer, target_population)`` tuples
        connection_args = self.parse_connection_params(connection_items)
        # Build Connection objects
        connections = []
        for (conn_model, src_lyr, src_pop, tgt_lyr, tgt_pop) in connection_args:
            model = self.connection_models[conn_model]
            source = self.layers[src_lyr]
            target = self.layers[tgt_lyr]
            connections.append(
                CONNECTION_TYPES[model.type](
                    model,
                    source, src_pop,
                    target, tgt_pop
                )
            )
        return connections

    def parse_connection_params(self, connection_items):
        """Return list of tuples specifying all unique connections

        Args:
            connection_items: Content of the ``connections``
                network/topology parameter. See ``self.build_connections`` for
                detailed description.

        Return:
            list: List of unique (<connection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>) tuples specifying all connections in
                the network.
        """
        connection_args = []
        for connection_item in connection_items:
            for source_layer_name, target_layer_name in itertools.product(
                connection_item['source_layers'],
                connection_item['target_layers']
            ):
                connection_args.append(
                    (
                        connection_item['connection_model'],
                        source_layer_name,
                        connection_item['source_population'],
                        target_layer_name,
                        connection_item['target_population'],
                    )
                )

        # Check that there are no duplicates.
        if not len(set(connection_args)) == len(connection_args):
            raise ParameterError(
                """Duplicate connections specified by `connections` topology
                parameter. (<connection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>) tuples should uniquely specify
                connections."""
            )

        return sorted(set(connection_args))

    def build_recorders(self, recorders_tree):
        """Build PopulationRecorder and ConnectionRecorder objects.

        Validates the ``recorders`` parameter tree and calls
        ``Network.build_population_recorders`` and
        ``Network.build_connection_recorders``

        Args:
            self (``Network``): Network object
            recorders_tree (``ParamsTree``): ``ParamsTree`` object without
                children nor ``nest_params``.
                The parameters of which may contain a ``population_recorders``
                (default []) and a ``connection_recorders`` (default []) entry
                specifying the network's recorders.
                The ``population_recorders`` and ``connection_recorders``
                entries are passed to (respectively)
                ``Network.build_population_recorders`` and
                ``Network.build_connection_recorders``

        Returns:
            (list(PopulationRecorder), list(ConnectionRecorder))
        """

        OPTIONAL_RECORDERS_PARAMS = {
            'population_recorders': [],
            'connection_recorders': [],
        }

        # Validate recorders tree
        # No children
        validation.validate_children(
            recorders_tree, [], []
        )
        # No nest_params
        validation.validate(
            'recorders', dict(recorders_tree.nest_params),
            param_type='nest_params',
            mandatory=[], optional={}
        )
        # Only a 'population_params' or 'connection_params' `params` entry
        recorders_params = validation.validate(
            'recorders', dict(recorders_tree.params), param_type='params',
            mandatory=[], optional=OPTIONAL_RECORDERS_PARAMS
        )

        return (
            self.build_population_recorders(
                recorders_params['population_recorders']
            ),
            self.build_connection_recorders(
                recorders_params['connection_recorders']
            )
        )

    def build_connection_recorders(self, connection_recorders_items):
        """Return connection recorders specified by a list of recorder params.

        ConnectionRecorders must be built after Connections.

        Arguments:
            connection_recorders_items (list | None): Content of the
                ``connection_recorders`` network/recorders parameter. A list of
                items describing the connection recorders to be created. Each
                item must be a ``dict`` of the following form::
                    dict: {
                        'model': <recorder_model>
                        'connection_model' : <connection_model>,
                        'source_layers': <source_layers_list>,
                        'source_population': <source_population>,
                        'target_layers': <target_layers_list>,
                        'target_population': <target_population>,
                    }
                Where <model> is the name of the connection recorder model (eg
                'weight_recorder'). The other keys fully specify the list of
                population-to-population connections of a certain model that
                a connection recorder is created for. Refer to
                `Network.build_connections` for a full description of how
                the <connection_model>, <source_layers>, <source_population>,
                <target_layers>, <target_population> keys are interpreted.

        Returns:
            list: List of ``ConnectionRecorder`` objects.
        """
        if connection_recorders_items is None:
            connection_recorders_items = []
        # Get all unique ``(model, connection_model, source_layer,
        # source_population, target_layer, target_population)`` tuples
        conn_recorder_args = []
        for item in connection_recorders_items:
            item = dict(item)  # TODO Fix this
            model = item.pop('model')
            conn_recorder_args += [
                (model,) + conn_args
                for conn_args in self.parse_connection_params([item])
            ]
        conn_recorder_args = sorted(conn_recorder_args)

        # Check that there are no duplicates.
        if not len(set(conn_recorder_args)) == len(conn_recorder_args):
            raise ParameterError(
                """Duplicate connection recorders specified by
                ``connection_recorders`` network/recorders parameter.
                (<recorder_model>, <connection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>) tuples should uniquely specify
                connections and connection recorders."""
            )

        connection_recorders = []
        for (
            model, conn_model, src_layer, src_pop, tgt_layer, tgt_pop
        ) in conn_recorder_args:

            matching_connections = [
                c for c in self.connections
                if (c.model.name == conn_model
                    and c.source.name == src_layer
                    and c.source_population == src_pop
                    and c.target.name == tgt_layer
                    and c.target_population == tgt_pop)
            ]

            # Check that all the connections exist in the network
            if not any(matching_connections):
                raise ParameterError(
                    f"Could not create connection recorder {model} for"
                    f" connection `{conn_model}, {src_layer}, {src_pop},"
                    f" {tgt_layer}, {tgt_pop}`: Connection does not exist in "
                    f"the network."
                )
            # Check (again) that connections are unique
            if len(matching_connections) > 1:
                raise ParameterError("Multiple identical connections")

            connection = matching_connections[0]
            # Create connection recorder
            connection_recorders.append(
                ConnectionRecorder(model, connection)
            )

        return connection_recorders

    def build_population_recorders(self, population_recorders_items):
        """Return population recorders specified by a list of recorder params.

        Arguments:
            population_recorders_items (list | None): Content of the
                ``population_recorders`` network/recorders parameter. A list of
                items describing the population recorders to be created and
                connected to the network. Each item must be a ``dict`` of the
                following form::
                    dict: {
                        'model' : <model>,
                        'layers': <layers_list>,
                        'populations': <populations_list>,
                    }
                Where:
                    - <model> is the model of a recorder.
                    - <layers_list> is None or a list of layer names. If
                      ``None``, all the layers in the network are considered.
                    - <populations_list> is None or a list of populations. If
                      ``None``, all the populations in each layer of interest
                      are considered. For ``InputLayer`` layers, only the
                      population of parrot neurons can be recorded.
                For each item in the list, a recorder of ``model`` will be
                created and connected to the population(s) of interest of each
                layer(s) of interest.

        Returns:
            list: List of ``PopulationRecorder`` objects.
        """
        if population_recorders_items is None:
            population_recorders_items = []
        # Get all (model, layer_name, population_name) tuples
        population_recorders_args = []
        # Iterate on layers x population for each item in list
        for item in population_recorders_items:
            model = item['model']
            layer_names = item['layers']
            # Use all layers if <layers> is None
            if layer_names is None:
                layer_names = self.layers.keys()
            for layer_name in layer_names:
                layer = self.layers[layer_name]
                population_names = item['populations']
                # Use all recordable population in layer if <population> is None
                if population_names is None:
                    population_names = layer.recordable_population_names()
                # Otherwise use only the populations specified if they exist in
                # the layer
                for population_name in [
                    p for p in population_names
                    if p in layer.recordable_population_names()
                ]:
                    population_recorders_args.append(
                        (model, layer_name, population_name)
                    )

        # Build the unique population recorder objects
        return [
            PopulationRecorder(
                model,
                layer=self.layers[layer_name],
                population_name=population_name
            )
            for (model, layer_name, population_name)
            in sorted(set(population_recorders_args))
        ]

    def __repr__(self):
        return '{classname}({tree})'.format(
            classname=type(self).__name__, tree=(self.tree))

    def __str__(self):
        return repr(self)

    @staticmethod
    def _create_all(objects):
        for obj in tqdm(objects):
            obj.create()

    def _layer_call(self, method_name, *args, layer_type=None, **kwargs):
        """Call a method on each layer."""
        for layer in self._get_layers(layer_type=layer_type):
            method = getattr(layer, method_name)
            method(*args, **kwargs)

    def _layer_get(self, attr_name, layer_type=None):
        """Get an attribute from each layer."""
        return tuple(
            getattr(layer, attr_name)
            for layer in self._get_layers(layer_type=layer_type))

    def _get_layers(self, layer_type=None):
        """Return all layers of a certain type ('Layer' or 'InputLayer')"""
        if layer_type is None:
            return sorted(self.layers.values())
        return [
            l for l in sorted(self.layers.values())
            if type(l).__name__ == layer_type
        ]

    def recorder_call(self, method_name, *args, recorder_class=None,
                      recorder_type=None, **kwargs):
        """Call a method on all recorder objects

        Args:
            method_name (str): Name of method of recorder objects.
            recorder_class, recorder_type (str or None): Passed to
                self.get_recorders()
            *args: Passed to method ``method_name``

        Kwargs:
            **kwargs: Passed to method ``method_name``
        """
        for recorder in self.get_recorders(
            recorder_class=recorder_class,
            recorder_type=recorder_type
        ):
            method = getattr(recorder, method_name)
            method(*args, **kwargs)

    def get_recorders(self, recorder_class=None, recorder_type=None):
        """Yield all ``PopulationRecorder`` and ``ConnectionRecorder`` objects.

        Args:
            recorder_class (str or None): Class of queried recorders.
                "PopulationRecorder", "ConnectionRecorder" or None.
            recorder_type (str or None): Type of queried recorders.
                'multimeter', 'spike_detector' or 'connection_recorder'
        """
        if recorder_type in ['multimeter', 'spike_detector']:
            recorder_class = 'PopulationRecorder'
        elif recorder_type in ['weight_recorder']:
            recorder_class = 'ConnectionRecorder'
        elif recorder_type is not None:
            raise ValueError('Recorder type not recognized')
        if recorder_class == 'PopulationRecorder' or recorder_class is None:
            yield from self.get_population_recorders(
                recorder_type=recorder_type
            )
        if recorder_class == 'ConnectionRecorder' or recorder_class is None:
            yield from self.get_connection_recorders(
                recorder_type=recorder_type
            )

    def get_population_recorders(self, recorder_type=None):
        """Yield ``PopulationRecorder`` objects of type ``recorder_type``."""
        if recorder_type not in [
            "multimeter", "spike_detector", None
        ]:
            raise ValueError('Recorder type not recognized')
        return iter([
            poprec for poprec in self.population_recorders
            if recorder_type is None or poprec.type == recorder_type
        ])

    def get_connection_recorders(self, recorder_type=None):
        """Yield ``ConnectionRecorder`` objects of type ``recorder_type``."""
        if recorder_type not in [
            "weight_recorder", None
        ]:
            raise ValueError('Unrecognized recorder type')
        yield from self.connection_recorders

    def _get_synapses(self, synapse_type=None):
        """Return synapse models"""
        if synapse_type is None:
            return sorted(self.synapse_models.values())
        return [
            syn for syn in sorted(self.synapse_models.values())
            if syn.type == synapse_type
        ]

    @if_not_created
    def create(self):
        # TODO: use progress bar from PyPhi?
        log.info('Creating neuron models...')
        self._create_all(self.neuron_models.values())
        log.info('Creating synapse models...')
        self._create_all(self.synapse_models.values())
        log.info('Creating recorder models...')
        self._create_all(self.recorder_models.values())
        log.info('Creating layers...')
        self._create_all(self._get_layers())
        log.info('Creating population recorders...')
        self._create_all(self.population_recorders)
        log.info('Creating connection recorders...')
        # ConnectionRecorders must be created BEFORE Connections
        self._create_all(self.connection_recorders)
        log.info('Connecting layers...')
        self._create_all(self.connections)
        self.print_network_size()

    @staticmethod
    def change_synapse_states(synapse_changes):
        """Change parameters for some connections of a population.

        Args:
            synapse_changes (list): List of dictionaries each of the form::
                    {
                        'synapse_model': <synapse_model>,
                        'params': {<param1>: <value1>}
                    }
                where the dictionary in ``params`` is passed to nest.SetStatus
                to set the parameters for all connections with synapse model
                ``<synapse_model>``
        """
        import nest
        for changes in tqdm(
                sorted(synapse_changes, key=synapse_sorting_map),
                desc="-> Changing synapses's state."):
            target_conns = nest.GetConnections(
                synapse_model=changes['synapse_model']
            )
            change_params = changes['params']
            log.info("Changing status for %s connections of type %s. Applying dict: %s", len(target_conns), changes['synapse_model'], change_params)
            nest.SetStatus(target_conns, change_params)

    def change_unit_states(self, unit_changes):
        """Change parameters for some units of a population.

        Args:
            unit_changes (list): List of dictionaries each of the form::
                    {
                        'layers': <layer_name_list>,
                        'layer_type': <layer_type>,
                        'population': <pop_name>,
                        'change_type': <change_type>,
                        'proportion': <prop>,
                        'params': {<param_name>: <param_value>,
                                   ...}
                    }
                where:
                ``<layer_name_list>`` (default None) is the list of name of the
                    considered layers. If not specified or empty, changes are
                    applied to all the layers of type <layer_type>.
                ``<layer_type>`` (default None) is the name of the type of
                    layers to which the changes are applied. Should be 'Layer'
                    or 'InputLayer'. Used only if <layer_name> is None.
                ``<population_name>`` (default None) is the name of the
                    considered population in each layer. If not specified,
                    changes are applied to all the populations.
                ``<change_type>`` ('constant' or 'multiplicative'). If
                    'multiplicative', the set value for each parameter is the
                    product between the preexisting value and the given value.
                    If 'constant', the given value is set without regard for the
                    preexisting value. (default: 'constant')
                ``<prop>`` (default 1) is the proportion of units of the
                    considered population on which the filter is applied. The
                    changes are applied on the units that are randomly selected
                    and passed the filter.
                ``'params'`` (default {}) is the dictionary of parameter changes
                    applied to the selected units.
        """
        for changes in sorted(unit_changes, key=unit_sorting_map):
            # Pass if no parameter dictionary.
            if not changes['params']:
                continue

            # Iterate on all layers of a given subtype or on a specific layer
            change_layers = changes.get('layers', [])
            if not change_layers:
                layers = self._get_layers(
                    layer_type=changes.get('layer_type', None))
            else:
                layers = [self.layers[layer_name]
                          for layer_name in change_layers]

            log.debug('    Applying unit changes dictionary %s\n\nto layers\n%s', changes, change_layers)
            for layer in tqdm(layers, desc="---> Apply change dict on layers"):
                layer.change_unit_states(
                    changes['params'],
                    population=changes.get('population', None),
                    proportion=changes.get('proportion', 1.),
                    change_type=changes.get('change_type', 'constant')
                )

    @staticmethod
    def reset():
        """Call `nest.ResetNetwork()`"""
        import nest
        nest.ResetNetwork()

    def save_metadata(self, output_dir):
        """Save network metadata.

            - Save recorder metadata
        """
        # Save recorder metadata
        self.recorder_call('save_metadata', output_dir)

    @staticmethod
    def print_network_size():
        import nest
        log.info('Network size (including recorders and parrot neurons):\n'
                 'Number of nodes: %s\n'
                 'Number of connections: %s',
                 nest.GetKernelStatus('network_size'),
                 nest.GetKernelStatus('num_connections')
                 )

    def populations_by_gids(self, layer_type='Layer'):
        """Return a dict of the form ``{'gid': (layer_name, pop_name)}``."""
        all_pops = {}
        for layer in self._get_layers(layer_type=layer_type):
            all_pops.update({
                gid: (layer.name, population)
                for gid, population in layer.populations.items()
            })
        return all_pops


def unit_sorting_map(unit_change):
    """Map by (layer, population, proportion, params_items for sorting."""
    return (unit_change.get('layers', 'None'),
            unit_change.get('layer_type', 'None'),
            unit_change.get('population', 'None'),
            unit_change.get('proportion', '1'),
            sorted(unit_change['params'].items()))


def synapse_sorting_map(synapse_change):
    """Map by (synapse_model, params_items) for sorting."""
    return (synapse_change['synapse_model'],
            sorted(synapse_change['params'].items()))
