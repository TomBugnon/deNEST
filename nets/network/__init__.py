#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/__init__.py
"""Provide a class to construct a network from independent parameters."""

import os
import itertools

from tqdm import tqdm

from .connections import (ConnectionModel, FromFileConnection,
                          MultiSynapseConnection, RescaledConnection,
                          TopoConnection)
from .layers import InputLayer, Layer
from .models import Model, SynapseModel
from .populations import Population
from .utils import if_not_created, log

# pylint: disable=too-few-public-methods

LAYER_TYPES = {
    None: Layer,
    'InputLayer': InputLayer,
}

CONNECTION_TYPES = {
    'topological': TopoConnection,
    'rescaled': RescaledConnection,
    'from_file': FromFileConnection,
    'multisynapse': MultiSynapseConnection,
}


class Network:
    """Represent a full network."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, params):
        """Initialize the network object without creating it in NEST."""
        self._created = False
        self._changed = False
        self.params = params
        # Build network components
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.neuron_models = self.build_named_leaves_dict(
            Model, self.params.c['neuron_models'])
        self.synapse_models = self.build_named_leaves_dict(
            SynapseModel, self.params.c['synapse_models'])
        self.recorder_models = self.build_named_leaves_dict(
            Model, self.params.c['recorder_models'])
        # Layers can have different types
        self.layers = {
            name: LAYER_TYPES[leaf['type']](name, leaf)
            for name, leaf in self.params.c['layers'].named_leaves()
        }
        self.connection_models = self.build_named_leaves_dict(
            ConnectionModel, self.params.c['connection_models'])
        # Connections must be built last
        conn_nested_list = [
            self.build_connection(connection_item)
            for connection_item in self.params.c['topology']['connections']
        ]
        self.connections = self.sort_connections([
            conn for sublist in conn_nested_list for conn in sublist
        ]) # .flatten() and sort to put "MultiSynapseConnection" connections at
            # the end
        # Populations are represented as a list
        self.populations = sorted(
            self.build_named_leaves_list(self.build_population,
                                         self.params.c['populations']))

    @staticmethod
    def build_named_leaves_dict(constructor, node):
        return {
            name: constructor(name, leaf)
            for name, leaf in node.named_leaves()
        }

    @staticmethod
    def sort_connections(connection_list):
        """Sort connections making sure that the `multisynapse` are last."""
        return (sorted([conn for conn in connection_list
                        if not conn.params['type'] == 'multisynapse'])
                + sorted([conn for conn in connection_list
                        if conn.params['type'] == 'multisynapse']))

    @staticmethod
    def build_named_leaves_list(constructor, node):
        return [constructor(name, leaf) for name, leaf in node.named_leaves()]

    def build_connection(self, connection_dict):
        """Return list of connections for source x target layer combinations."""
        source_layers = [self.layers[layer]
                         for layer in connection_dict['source_layers']]
        target_layers = [self.layers[layer]
                         for layer in connection_dict['target_layers']]
        model = self.connection_models[connection_dict['connection']]
        return [
            CONNECTION_TYPES[model.type](source, target, model, connection_dict)
            for source, target
            in itertools.product(source_layers, target_layers)
        ]

    def build_population(self, pop_name, pop_params):
        # Get the gids and locations for the population from the layer object.
        layer = self.layers[pop_params['layer']]
        return Population(pop_name, layer, pop_params)

    def __repr__(self):
        return '{classname}({params})'.format(
            classname=type(self).__name__, params=(self.params))

    def __str__(self):
        return repr(self)

    @staticmethod
    def _create_all(objects):
        for obj in tqdm(objects):
            obj.create()

    def _layer_call(self, method_name, *args, layer_type=None, **kwargs):
        """Call a method on each input layer."""
        for layer in self._get_layers(layer_type=layer_type):
            method = getattr(layer, method_name)
            method(*args, **kwargs)

    def _layer_get(self, attr_name, layer_type=None):
        """Get an attribute from each layer."""
        return tuple(
            getattr(layer, attr_name)
            for layer in self._get_layers(layer_type=layer_type))

    def _get_layers(self, layer_type=None):
        if layer_type is None:
            return sorted(self.layers.values())
        return [
            l for l in sorted(self.layers.values())
            if type(l).__name__ == layer_type
        ]

    def recorder_call(self, method_name, *args, recorder_class=None,
                       recorder_type=None, **kwargs):
        """Call a method on population and/or connection recorders.

        Args:
            method_name (str): Name of method of Recorder objects.
            recorder_class (str or None): Class of recorders: "population",
                "connection" or None. Passed to self.get_recorders()
            recorder_type (str or None): Passed to self.get_recorders()
        """
        for recorder in self.get_recorders(
            recorder_class=recorder_class,
            recorder_type=recorder_type):
            method = getattr(recorder, method_name)
            method(*args, **kwargs)

    def get_recorders(self, recorder_class=None, recorder_type=None):
        """Generator to get each pop and/or conn recorder of a certain type."""
        assert recorder_class in ["population", "connection", None], \
            "Unrecognized recorder class"
        if recorder_class == 'population' or recorder_class is None:
            for population in self.populations:
                yield from population.get_recorders(recorder_type=recorder_type)
        if recorder_class == 'connection' or recorder_class is None:
            for connection in self.connections:
                yield from connection.get_recorders(recorder_type=recorder_type)

    def _get_synapses(self, synapse_type=None):
        if synapse_type is None:
            return sorted(self.synapse_models.values())
        return [
            syn for syn in sorted(self.synapse_models.values())
            if syn.type == synapse_type
        ]

    @property
    def any_inputlayer(self):
        return bool(self._get_layers(layer_type='InputLayer'))

    @property
    def input_shapes(self):
        return set(self._layer_get('shape', layer_type='InputLayer'))

    @property
    def max_input_shape(self):
        """Max of each dimension."""
        return (max([s[0] for s in self.input_shapes]),
                max([s[1] for s in self.input_shapes]))

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
        log.info('Connecting layers...')
        self._create_all(self.connections)
        self.print_network_size()
        log.info('Creating recorders...')
        self._create_all(self.populations)

    def dump_connections(self, output_dir):
        for connection in tqdm(self.connections, desc='Dumping connections'):
            connection.dump(output_dir)

    @staticmethod
    def change_synapse_states(synapse_changes):
        """Change parameters for some connections of a population.

        Args:
            synapse_changes (list): List of dictionaries each of the form::
                    {
                        'synapse_model': <synapse_model>,
                        'params': {<key>: <value>,
                                    ...}
                    }
                where the ``'params'`` key contains the parameters to set for
                all synapses of a given model.
        """
        import nest
        for changes in tqdm(
                sorted(synapse_changes, key=synapse_sorting_map),
                desc="-> Changing synapses's state."):
            target_conns = nest.GetConnections(
                synapse_model=changes['synapse_model']
            )
            change_params = changes['params']
            print(f"Change status for N={len(target_conns)} conns of type "
                  f"{changes['synapse_model']}. Apply dict: {change_params}")
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
                        'filter': <filter>,
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
                ``filter`` (default {}) is a dictionary defining the filter
                    applied onto the proportion of randomly selected units of
                    the population. The filter defines an interval for any unit
                    parameter. A unit is selected if all its parameters are
                    within their respectively defined interval. The parameter
                    changes are applied only on the selected units.
                    The ``filter`` dictionary is of the form:
                        {
                            <unit_param_name_1>:
                                'min': <float_min>
                                'max': <float_max>
                            <unit_param_name_2>:
                                ...
                        }
                    Where <float_min> and <float_max> define the (inclusive)
                    min and max of the filtering interval for the considered
                    parameter (default resp. -inf and +inf)
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

            # Verbose
            print(f'\n--> Applying unit changes dictionary: {changes} ... to'
                  f' layers: {change_layers}')

            for layer in tqdm(layers, desc="---> Apply change dict on layers"):
                layer.change_unit_states(
                    changes['params'],
                    population=changes.get('population', None),
                    proportion=changes.get('proportion', 1.),
                    filter_dict=changes.get('filter', {}),
                    change_type= changes.get('change_type', 'constant')
                )
            print(f'\n')

    @staticmethod
    def reset():
        import nest
        nest.ResetNetwork()

    def set_input(self, stimulus_array, start_time=0.):
        self._layer_call('set_input', stimulus_array, start_time,
                         layer_type='InputLayer')

    def save_metadata(self, output_dir):
        """Save network metadata."""
        # Save recorder metadata
        self.recorder_call('save_metadata', output_dir)

    def plot_connections(self, output_dir):
        for conn in tqdm(self.connections, desc='Creating connection plots'):
            conn.save_plot(output_dir)

    @staticmethod
    def print_network_size():
        import nest
        print('------------------------')
        print('Network size (without recorders)')
        print('Number of nodes: ', nest.GetKernelStatus('network_size'))
        print('Number of connections: ',
              nest.GetKernelStatus('num_connections'))
        print('------------------------')

    def populations_by_gids(self, layer_type='Layer'):
        """Return a dictionary of the form {'gid': (layer_name, pop_name)}."""
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
