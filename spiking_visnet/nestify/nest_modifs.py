#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nest_modifs.py

"""Gather and modify NEST and network parameters."""

import random

import nest
import numpy as np
from tqdm import tqdm


def change_synapse_states(synapse_changes):
    """Change synapse status by synapse models.

    Args:
        network (Network object): Initialized network
        synapse_changes (list): List of dictionaries each of the form::
                {synapse_model: <synapse_model>,
                 params: {<key>: <value>,
                          ...}}
            where the params contains the parameters to set for all synapses of
            a given model.

    """
    for changes in tqdm(sorted(synapse_changes, key=synapse_sorting_map),
                        desc="-> Change synapses's state."):
        nest.SetStatus(
            nest.GetConnections(synapse_model=changes['synapse_model']),
            changes['params']
            )


def synapse_sorting_map(synapse_change):
    """Map by (synapse_model, params_items) for sorting."""
    return (synapse_change['synapse_model'],
            sorted(synapse_change['params'].items()))


def change_unit_states(unit_changes, network_layers):
    """Change parameters for some units of a population.

    Args:
        unit_changes (list): List of dictionaries each of the form::
                {'layer': <layer_name>,
                 'population': <pop_name>,
                 'proportion': <prop>,
                 'params': {<param_name>: <param_value>,
                            ...}
                }
            where <layer_name> and <population_name> define the considered
            population, <prop> is the proportion of units of that population
            for which the parameters are changed, and the ``'params'`` entry is
            the dictionary of parameter changes apply to the selected units.

    """
    for changes in tqdm(sorted(unit_changes, key=unit_sorting_map),
                        desc="-> Change units' state"):
        layer_gid = network_layers[changes['layer']]['gid']
        all_pop_units = np.array(
            [nd for nd in nest.GetLeaves(layer_gid)[0]
             if nest.GetStatus([nd], 'model')[0] == changes['population']]
        )
        num_neurons = len(all_pop_units)
        num_to_change = int(changes['proportion'] * num_neurons)

        gids_to_change = np.random.randint(num_neurons,
                                           size=(1, num_to_change))[0]

        nest.SetStatus(all_pop_units[gids_to_change].tolist(),
                       params=changes['params'])

def unit_sorting_map(unit_change):
    """Map by (layer, population, proportion, params_items for sorting."""
    return (unit_change['layer'],
            unit_change['population'],
            unit_change['proportion'],
            sorted(unit_change['params'].items()))
