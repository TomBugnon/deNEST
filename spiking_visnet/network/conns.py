#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# format_recorders.py

"""Query, filter and modify specific synaptic connections from NEST."""


import nest


def get_filtered_synapses(gid, ref='source', secondary_layer_gid=None,
                          secondary_population=None, synapse_model=None):
    """Return a unit's synapses that come from or target a specific population.

    Args:
        gid (int): GID of the reference unit.
        network (Network): Initialized network object.
        ref (str): Determines whether we query the pre-synaptic (ref=='target')
            or post_synaptic (ref=='source') units.
        secondary_layer_gid (int): Layer gid of the queried synapses.
        secondary_population (str): Population name of the queried synapses.

    Return:
        list: List of dictionaries containing the incoming/outgoing connections
            for the reference unit (GID), filtered by layer and population. Each
            dictionary is the output of nest.GetStatus() applied on a
            connection.

    """
    if not isinstance(secondary_layer_gid, int):
        Warning("""'secondary_layer_gid' argument should be an integer rather
                than a sequence. Try using the first element""")
        secondary_layer_gid = secondary_layer_gid[0]

    if ref == 'source':
        all_synapses = nest.GetStatus(nest.GetConnections(source=(gid,)))
    elif ref == 'target':
        all_synapses = nest.GetStatus(nest.GetConnections(target=(gid,)))

    filter_key = {'source': 'target',
                  'target': 'source'}

    return [synapse for synapse in all_synapses
            if (secondary_layer_gid is None or
                layer(synapse[filter_key[ref]]) == secondary_layer_gid)
            and (not secondary_population or
                 model(synapse[filter_key[ref]]) == secondary_population)
            and (not synapse_model or
                 synapse['synapse_model'] == synapse_model)]


def layer(gid):
    """Return the GID (int) of a unit's (int) layer."""
    return nest.GetStatus((gid,), 'parent')[0]


def model(gid):
    """Return the model (str) of a unit (int)."""
    return nest.GetStatus((gid,), 'model')[0]
