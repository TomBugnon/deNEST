#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""Provides the `Network` class."""

import collections

from tqdm import tqdm

from .nestify.format_net import get_network
from .nestify.init_nest import (gid_location_mapping, init_nest,
                                set_nest_savedir)
from .save import generate_save_subdir_str, get_nest_tmp_savedir
from .utils.structures import deepcopy_dict

STIM_LAYER_SUFFIX = '_stimulator'


# TODO: move functionality from nestify/format_net to this class


class Network(collections.UserDict):
    """Represents a neural network.

    Provides an interface to NEST and handles initialization of the NEST kernel.
    """

    def __init__(self, network_params, sim_params):
        """Initialize object from parameter tree."""
        super().__init__(get_network(network_params))
        # Introduce parrot layers between input stimulators and neurons
        self.introduce_parrot_layers()
        # Get the saving subdirectory. Output dir is OUTPUT_DIR/save_subdir_str.
        self.get_save_subdir_str(network_params, sim_params)

    def get_save_subdir_str(self, full_params_tree, param_file_path):
        """Record the subdirectory string for this network/simulation."""
        self.save_subdir_str = generate_save_subdir_str(full_params_tree,
                                                        param_file_path)

    def init_nest(self, kernel_params, sim_params):
        """Initialize NEST kernel and network.

        The GIDs of the created NEST objects are added in place to the Network
        object.
        Provides the NEST kernel with the path to the temporary directory where
        it will save the recorders.
        Add to the network object the mappings between layer location and GID.

        Args:
            params: Full parameter tree.
        """
        # Initialize NEST
        init_nest(self, kernel_params)
        # Tell NEST kernel where to save all the recorders.
        tmp_OUTPUT_DIR = get_nest_tmp_savedir(self, sim_params)
        set_nest_savedir(tmp_OUTPUT_DIR)
        # Get the bi-directional GID-location mappings for each population.
        self.get_gid_location_mappings()

    def input_layer(self):
        """Return (name, nest_params, params) for an arbitrary input layer."""
        input_layer_name = self['areas'][self.input_area_name()][0]
        return (input_layer_name,
                self['layers'][input_layer_name]['nest_params'],
                self['layers'][input_layer_name]['params'])

    def input_area_name(self):
        """Return name of network's input area."""
        layer_names = list(self['layers'].keys())
        return self['layers'][layer_names[0]]['params']['input_area_name']

    def filters(self):
        """Return the 'filters' dictionary for an arbitrary input layer.

        If no filter is applied to the input layers, return -1
        """
        (_, _, params) = self.input_layer()
        return params.get('filters', -1)

    def input_res(self):
        """Return the resolution of the network's input layers."""
        (_, nest_params, _) = self.input_layer()
        return (nest_params['rows'], nest_params['columns'])

    def get_gid_location_mappings(self):
        """Create a self.location attribute with the location/GID mappings.

        Creates a self.location attribute containing a tree such that:
            self.gid_coo[<layer_name>][<population_name>] = <gid_coo_dict>
        where:
            <gid_coo_dict> is a dictionary of the form::
                    ``{'gid': <gid_by_location_array>,
                       'location': <location_by_gid_mapping>}``
        Finally:
        - <gid_by_location_array> (np-array) is a
            (``nrows``, ``ncols``, ``nelems``)-array where:
            - (``nrows``, ``ncols``) is the dimension of the layer
            - ``nelems`` is the number of units of the considered population at
                each location.
        - <location_by_gid_mapping> (dict) is dictionary of which keys are
            GIDs (int) and entries are (``row``, ``col``) location (tuple of
            int)

        """
        self.locations = {}
        for pop_dict in tqdm(self['populations'],
                             desc='Create GID/location mappings'):

            lay_name, pop_name = pop_dict['layer'], pop_dict['population']
            layer_gid = self['layers'][lay_name]['gid']

            # Create self.location[lay_name] dictionary if it doesn't exist.
            if lay_name not in self.locations:
                self.locations[lay_name] = {}

            self.locations[lay_name][pop_name] = gid_location_mapping(layer_gid,
                                                                      pop_name)

    def introduce_parrot_layers(self):
        """Introduce layers of parrot neurons between stim devices and neurons.

        For each input layer that should use parrot neurons:
            - duplicate the input layer dictionary under a different layer name
                (stim_layer_name = base_input_layer_name + STIM_LAYER_SUFFIX)
            - Modify the original layer dictionary to replace the elements with
                parrot_neuron.
                (parrot_layer_name = base_input_layer_name)
            - Add one-to-one topological connections to the network's
                connections list to connect each stimulator input layer to its
                associated parrot input layer.
            - Add the populations for the new layers in network['populations']
                (without recorders).

        """
        for input_layer_name in self['areas'][self.input_area_name()]:

            old_entry = self['layers'][input_layer_name]
            if old_entry['params']['with_parrot']:

                # Duplicate layer entry
                copy_entry = deepcopy_dict(old_entry)
                # Change old entry to parrot elements
                elem_list = old_entry['nest_params']['elements']
                parrot_elem_list = change_input_elementname(elem_list,
                                                            'parrot_neuron')
                old_entry['nest_params']['elements'] = parrot_elem_list
                # Create new 'stimulation device' layer
                stim_layer_name = input_layer_name + STIM_LAYER_SUFFIX
                self['layers'][stim_layer_name] = copy_entry

                # Add the stimulator population
                stimulator_population_name = elem_list[0]  # TODO: function
                self['populations'].append(
                    {'layer': stim_layer_name,
                     'population': stimulator_population_name,
                     'mm': {'record_pop': False},
                     'sd': {'record_pop': False}}
                )

                # Connect the 'stim' layer to the parrot layer
                self['connections'].append(
                    {'source_layer': stim_layer_name,
                     'target_layer': input_layer_name,
                     'source_population': stimulator_population_name,
                     'target_population': 'parrot_neuron',
                     'nest_params': one_to_one_connection(),
                     'params': one_to_one_connection()}
                )

    def input_stim_layers(self):
        """Return a list of the stimulator input layers.

        Returns:
            list: List of strings of the form eg:
                - 'input_layer_o2', or
                - 'input_layer_o2'+STIM_LAYER_SUFFIX if the layer has parrot
                    neurons

        """
        return [layer_name + STIM_LAYER_SUFFIX if self.has_parrots(layer_name)
                else layer_name
                for layer_name in self.input_layers()]

    def has_parrots(self, layer_name):
        """Return the layer params' 'with_parrot' entry."""
        return self['layers'][layer_name]['params'].get('with_parrot', False)

    def input_layers(self):
        """Return the list of input layer names (without STIM_LAYER_SUFFIX)."""
        return self['areas'][self.input_area_name()]

    def populations(self, layer_name):
        """Return the list of populations in a layer."""
        return [elem for elem in (self['layers'][layer_name]
                                  ['nest_params']['elements'])
                if isinstance(elem, str)]

    def stimulator_type(self, population_name):
        """Return the type of stimulator in an input layer.

        The type of stimulator is either the name of the base nest model used
        to define the population, or the population name itself.

        Returns:
            str: 'poisson_generator' or 'spike_generator'.

        """
        nest_models = [nest_model
                       for (nest_model, network_model, _)
                       in self['neuron_models']
                       if network_model == population_name]
        return nest_models[0] if nest_models else population_name

        # TODO
    def save(self, path):
        """Save object."""
        pass


def change_input_elementname(elem_list, new_name):
    """Change all element names (strings) in list to <new_name>."""
    assert(len(elem_list) == 2), 'Input layers should not be composite.'
    return [new_name if isinstance(elem, str) else elem
            for elem in elem_list]


def one_to_one_connection(synapse_model='static_synapse'):
    """Return a connection dictionary used to connect 2 layers one-to-one.

    Similarly to the other layers, we use the nest.topology module for
    connecting layers, rather than nest.Connect. In order to assure that the
    connections are one-to-one, we set the number of connections by unit to
    one, and the radius of the mask to an infinitesimal value.

    NB: The returned dictionary can be used only to connect non-composite layers
    of equal size.

    """
    MASK_RADIUS = 0.0001
    return {"connection_type": "divergent",
            "mask": {"circular": {"radius": MASK_RADIUS}},
            'number_of_connections': 1,
            "synapse_model": synapse_model
            }
