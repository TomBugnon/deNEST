#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/network.py

"""Provide a class to construct a network from independent parameters."""

import os
import random

from tqdm import tqdm

from .connections import (ConnectionModel, FromFileConnection,
                          RescaledConnection, TopoConnection)
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
    'from_file': FromFileConnection
}


class Network:

    def __init__(self, params):
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
            Model, self.params.c['recorders'])
        # Layers can have different types
        self.layers = {
            name: LAYER_TYPES[leaf['type']](name, leaf)
            for name, leaf in self.params.c['layers'].named_leaves()
        }
        self.connection_models = self.build_named_leaves_dict(
            ConnectionModel, self.params.c['connection_models'])
        # Connections must be built last
        self.connections = sorted(
            [self.build_connection(connection)
             for connection in self.params.c['topology']['connections']]
        )
        # Populations are represented as a list
        self.populations = sorted(
            self.build_named_leaves_list(
                self.build_population,
                self.params.c['populations']
                )
            )

    @staticmethod
    def build_named_leaves_dict(constructor, node):
        return {name: constructor(name, leaf)
                for name, leaf in node.named_leaves()}

    @staticmethod
    def build_named_leaves_list(constructor, node):
        return [constructor(name, leaf)
                for name, leaf in node.named_leaves()]

    def build_connection(self, params):
        source = self.layers[params['source_layer']]
        target = self.layers[params['target_layer']]
        model = self.connection_models[params['connection']]
        return CONNECTION_TYPES[model.type](source, target, model, params)

    def build_population(self, pop_name, pop_params):
        # Get the gids and locations for the population from the layer object.
        layer = self.layers[pop_params['layer']]
        return Population(pop_name, layer, pop_params)

    def __repr__(self):
        return '{classname}({params})'.format(
            classname=type(self).__name__, params=(self.params))

    def __str__(self):
        return repr(self)

    def _create_all(self, objects):
        for obj in tqdm(objects):
            obj.create()

    def _layer_call(self, method_name, *args, layer_type=None, **kwargs):
        """Call a method on each input layer."""
        for layer in self._get_layers(layer_type=layer_type):
            method = getattr(layer, method_name)
            method(*args, **kwargs)

    def _layer_get(self, attr_name, layer_type=None):
        """Get an attribute from each layer."""
        return tuple(getattr(layer, attr_name)
                     for layer in self._get_layers(layer_type=layer_type))

    def _get_layers(self, layer_type=None):
        if layer_type is None:
            return sorted(self.layers.values())
        return [l for l in sorted(self.layers.values())
                if type(l).__name__ == layer_type]

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
        for connection in tqdm(self.connections,
                               desc='Dump connections'):
            connection.dump(output_dir)

    def change_synapse_states(self, synapse_changes):
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
        for changes in tqdm(sorted(synapse_changes, key=synapse_sorting_map),
                            desc="-> Change synapses's state."):
            nest.SetStatus(
                nest.GetConnections(synapse_model=changes['synapse_model']),
                changes['params']
            )

    def change_unit_states(self, unit_changes):
        """Change parameters for some units of a population.

        Args:
            unit_changes (list): List of dictionaries each of the form::
                    {
                        'layer': <layer_name>,
                        'population': <pop_name>,
                        'proportion': <prop>,
                        'params': {<param_name>: <param_value>,
                                   ...}
                    }
                where ``<layer_name>`` and ``<population_name>`` define the
                considered population, ``<prop>`` is the proportion of units of
                that population for which the parameters are changed, and the
                ``'params'`` entry is the dictionary of parameter changes apply
                to the selected units.
        """
        import nest
        for changes in tqdm(sorted(unit_changes, key=unit_sorting_map),
                            desc="-> Change units' state"):

            if self._changed and changes['proportion'] != 1:
                raise Exception("Attempting to change probabilistically some "
                                "units' state multiple times.")

            layer = self.layers[changes['layer']]
            all_gids = layer.gids(population=changes.get('population', None))
            gids_to_change = [all_gids[i] for i
                              in sorted(
                                  random.sample(range(len(all_gids)),
                                                int(len(all_gids)
                                                    * changes['proportion'])
                                                ))]
            nest.SetStatus(gids_to_change,
                           params=changes['params'])
        self._changed = True

    def reset(self):
        import nest
        nest.ResetNetwork()

    def set_input(self, stimulus, start_time=0.):
        self._layer_call('set_input', stimulus, start_time,
                         layer_type='InputLayer')

    def save(self, output_dir, with_rasters=True):
        # Save synapses
        for conn in self.connections:
            conn.save(output_dir)
        # Save recorders
        for population in tqdm(self.populations,
                               desc='Save formatted recorders'):
            population.save(output_dir, with_rasters=with_rasters)

    def plot_connections(self, output_dir):
        for conn in tqdm(self.connections,
                         desc='Create connection plots'):
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

    def dump_connection_numbers(self, ratio_dump_dir):
        from ..save import save_as_yaml
        from os.path import join
        layer_connections = {}
        for layer in tqdm(self._get_layers(layer_type='Layer'),
                          desc='Dumping connection numbers per layer'):
            layer_connections[layer.name] = layer.incoming_connections()
        save_as_yaml(join(ratio_dump_dir, 'synapse_numbers.yml'),
                     layer_connections)


def unit_sorting_map(unit_change):
    """Map by (layer, population, proportion, params_items for sorting."""
    return (unit_change['layer'],
            unit_change['population'],
            unit_change['proportion'],
            sorted(unit_change['params'].items()))


def synapse_sorting_map(synapse_change):
    """Map by (synapse_model, params_items) for sorting."""
    return (synapse_change['synapse_model'],
            sorted(synapse_change['params'].items()))
