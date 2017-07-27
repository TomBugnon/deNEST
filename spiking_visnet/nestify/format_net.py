#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/format_net.py

"""Format NEST friendly network from the raw network tree."""

import copy as cp
from collections import ChainMap

from ..utils import filter_suffixes as filt
from ..utils import structures as struct


def get_network(raw_network_tree):
    """Format the raw network parameter tree.

    Returns:
        - (dict): Dictionary of the form

                {'neuron_models':<neuron_models>,
                 'synapse_models':<synapse_models>,
                 'layers':<layers>,
                 'non_expanded_layers': <non_expanded_layers>,
                 'connections':<connections>,
                 'areas':<areas>},

                 where:

            - <neuron_models> is list of tuples each of the form:
                (<base_nest_model>,<model_name>,<params_chainmap>)

            - <synapse_models> is list of tuples each of the form:
                (<base_nest_model>,<model_name>,<params_chainmap>)

            - <layers> is a dictionary of the form
                {<layer_name>: {'params': <params_chainmap>
                                'nest_params': <nest_params_chainmap>}
                 where
                 - 'params' contains all the parameters related to this layer,
                 - 'nest_params' contains the nest_formatted parameters
                    used to create the layer,

            - <non_expanded_layers> is similar to layers but without layer
                duplication for different filters.

            - <connections> is a list of tuples each of the form:
                (<source_layer>, <target_layer>, <params_chainmap>)

            - <areas> is a dictionary of the form:
                {<area_name>: <list_of_layers>} where <list_of_layers> is the
                list of all layers of the network within a given area

    """
    layers = get_layers(raw_network_tree['layers'], expanded=True)
    return {
        'neuron_models': get_models(raw_network_tree['neuron_models']),
        'synapse_models': get_models(raw_network_tree['synapse_models']),
        'layers': layers,
        'connections': get_connections(raw_network_tree),
        'areas': get_areas(layers),
        'populations': get_populations(raw_network_tree),
    }


def get_models(model_tree):
    """Return the leaf models in a model dictionary.

    Args:
        - <model_tree>: Parameter tree for models. Should contain at depth 0
            under the key `nest_model` the name of the base nest model in the
            tree inherits of.

    Returns:
        (list): List of tuples  of the form:
            (<base_nest_model, <model_name>, <params_chainmap)
            where each tuple describes a model that will be used in
            the network.

    """
    return struct.flatten([
        struct.distribute_to_tuple(
            struct.traverse(model_dict,
                            params_key='params',
                            children_key='children',
                            name_key='name',
                            accumulator=[]),
            model_dict['nest_model'],
            pos=0)
        for key, model_dict in model_tree.items()
    ])


def get_layers(layers_tree, expanded=True):
    """Generate from a tree a flat dictionnary describing the layers.

    Args:
        - <layers_tree> (dict): Tree that will be traversed to gather all
            parameters of each layer-leaf. The thus gathered parameters are
            then formatted to produce nest-readable parameter dictionaries.
        - <expanded> (bool): If true, returns the expanded tree after taking in
            account the replication of layers for different filters, otherwise
            don't replicate layers whatsoever.

    Returns:
        - (dict): Dictionary of the form:

            {<layer_name>: {'params': <params_chainmap>
                            'nest_params': <nest_params_chainmap>}

             where

             - 'params' contains all the parameters related to this layer,
             - 'nest_params' contains the nest_formatted parameters
                used to create the layer.

    """
    # List of tuples of the form (<layer_name>, <params_chainmap>).
    layer_list = save_base_name(struct.traverse(layers_tree,
                                                params_key='params',
                                                children_key='children',
                                                name_key='name',
                                                accumulator=[]))

    if expanded:
        # The layers whose <params_chainmap> contains the field 'filters' are
        # replicated with different names.
        return format_layer_list(expand_layer_list(layer_list))
    else:
        return format_layer_list(layer_list)


def save_base_name(layer_list):
    """Save input layer name before extension.

    Add the pre-extension layer name to the 'base_name' field of the
    parameters chainmap of each layer.

    Args:
        - <layer_list> (list): list of tuples of the form:
            (<layer_name>, <params_chainmap>).
    """
    return [(name, params_chainmap.new_child({'base_name': name}))
            for (name, params_chainmap) in layer_list]


def format_layer_list(layer_list):
    """Generate a layer dictionary from a layer list.

    Args:
        - <layer_list> (list): list of the form
            [(<layer_name>, <params_chainmap>), ...]

    Returns:
        (dict): Dictionary of the form
            {<layer_name>: {'params': <params_chainmap>,
                            'nest_params': <nest_params_chainmap>}

    """
    return {
        layer_name: {
            'params': params.new_child({'layer_name': layer_name}),
            'nest_params': format_nest_layer_params(params)
        }
        for (layer_name, params) in layer_list
    }


def format_nest_layer_params(layer_params):
    """Generate a nest-formatted parameter dict from a layer parameter dict."""
    nest_p = {}
    nest_p['rows'] = layer_params['size']
    nest_p['columns'] = layer_params['size']
    nest_p['extent'] = [layer_params['visSize'], layer_params['visSize']]
    nest_p['edge_wrap'] = layer_params['edge_wrap']
    nest_p['elements'] = get_layer_elements(layer_params)

    return nest_p


def get_layer_elements(layer_params):
    """Format for nest the list of elements of a layer.

    Returns:
        (dict): Dictionary of the form:
                {'elements': <elements_list>}
            where <elements_list> is eg of the form:
                ['L23_exc', 2, 'L23_inh', 1, 'L4_exc' , 2, ...]}

    """
    layer_elements = layer_params['elements']
    elements_list = []

    # Get number of inhibitory neurons (if any) in the layer (to multiply
    # with number of excitatory neurons.)
    inh_pop = [
        population['type'] == 'inhibitory' for population in layer_elements
    ]

    assert sum(inh_pop) <= 1, 'There should be only one inhibitory population'

    if sum(inh_pop) == 1:
        number_inh = layer_elements[inh_pop.index(True)]['ratio']

    # Build up element list
    for pop in layer_elements:
        # Number of populations in layer
        if pop['type'] == 'inhibitory':
            number = pop['ratio']
        elif pop['type'] == 'excitatory':
            if sum(inh_pop) == 0:
                number = pop['ratio']
            else:
                number = (pop['ratio'] * number_inh
                          * layer_params['exc_inh_ratio'])
        elements_list += [pop['population'], number]

    return elements_list


def get_areas(layer_dict):
    """Create an area dictionary from the layer dictionary.

    Invert the layer dictionary by reading the ['params']['area'] subkey of each
    layer to create an area dictionary.
    """
    return struct.invert_dict({layer: layer_params['params']
                               for (layer, layer_params) in layer_dict.items()},
                              inversion_key='area')


def get_connections(raw_network_tree):
    """Return the formatted connections of the network.

    Args:
        - <raw_network_tree>: Raw network tree (from the full param tree)

    Returns:
        (list) : list of tuples each of the form:
            (<source_layer>, <target_layer>, <nest-readable_params_chainmap>)
            where each tuple describes a connection after taking in account
            possible duplication of input layers.

    """
    exp_network_tree = expand_connections(raw_network_tree)
    exp_layers = get_layers(exp_network_tree['layers'], expanded=True)

    return [(conn['source_layer'],
             conn['target_layer'],
             get_conn_params(conn,
                             exp_network_tree['connection_models'],
                             exp_layers))
            for conn in exp_network_tree['connections']]


def get_conn_params(conn, connection_models, layers):
    """Generate nest readable connection parameters.

    Args:
        - conn (dict): Specific connection dictionary.
        - connection_models (dict): Models of connections
        - layers (dict): Expanded and formatted (flat) layer dictionnary.

    Return:
        (dict): NEST readable parameter dictionary for a given connection.

    """
    source_params = layers[conn['source_layer']]['params']
    target_params = layers[conn['target_layer']]['params']

    # Update connection models with the specific connection params (possibly
    # empty).
    conn_p = ChainMap(conn['params'],
                      connection_models[conn['connection']])

    # RF and weight scaling:
    # TODO: maskfactor?
    # TODO: btw error in ht files: secondary horizontal intralaminar mixes dcpS
    # and dcpP

    rf_factor = (target_params['rf_scale_factor'] * source_params['visSize'] /
                 (source_params['size'] - 1))
    return conn_p.new_child(
        {'sources': {'model': conn['source_population']},
         'targets': {'model': conn['target_population']},
         'mask': scaled_conn_mask(conn_p['mask'], rf_factor),
         'kernel': scaled_conn_kernel(conn_p['kernel'], rf_factor),
         'weights': scaled_conn_weights(conn_p['weights'],
                                        source_params['weight_gain'])})


def scaled_conn_mask(mask_dict, scale_factor):
    """Scale the size of a connection mask by `scale_factor`."""
    keys = list(mask_dict.keys())
    assert len(keys) <= 1, 'Wrong formatting of connection mask'
    mask_dict_copy = cp.deepcopy(mask_dict)

    if keys[0] == 'circular':
        mask_dict_copy['circular']['radius'] *= scale_factor
    elif keys[0] == 'rectangular':
        mask_dict_copy['rectangular'] = {
            key: [scale_factor * scalar for scalar in scalar_list]
            for key, scalar_list in mask_dict['rectangular'].items()
        }

    return mask_dict_copy


def scaled_conn_kernel(kernel, scale_factor):
    """Scale the size of a connection kernel by `scale_factor`."""
    if isinstance(kernel, (float, int)):
        return kernel
    elif isinstance(kernel, (dict)) and 'gaussian' in kernel.keys():
        kernel_copy = cp.deepcopy(kernel)
        kernel_copy['gaussian']['sigma'] *= scale_factor
    else:
        raise Exception('Wrong formatting of connection kernel')
    return kernel_copy


def scaled_conn_weights(weights, scale_factor):
    """Multiply connection weights by `scale_factor."""
    return (weights * scale_factor)


# def get_area_layer_params(area, params):
#     p = {}
#
#     p['rows'] = params['resolution']['size']['area']
#     p['columns'] = params['resolution']['size']['area']
#     p['extent'] = params['resolution']['visSize']
#     p['edge_wrap'] = params['resolution']['edge_wrap']
#
#     return p


def expand_layer_list(layer_list):
    """Duplicate with different names the layer tuples for different filters.

    Args:
        - <layer_list>: List of tuples each of the form
            (<layer_name>, <params_chainmap>)

    Returns:
        (list): List with the same format for which tuples have been duplicated
            with different <layer_name> if <params_chainmap> has a `filters`
            key.

    """
    expanded_list = []
    for (layer_name, params_chainmap) in layer_list:
        if 'filters' in params_chainmap.keys():
            expanded_list += [(ext_layer_name, params_chainmap)
                              for ext_layer_name
                              in filt.get_expanded_names(
                                  layer_name,
                                  params_chainmap['filters'])]
        else:
            expanded_list += [(layer_name, params_chainmap)]

    return expanded_list


def expand_connections(raw_network_tree):
    """Account for the duplication of input layers with different filters.

    Modifify the 'connections' subdictionary of <raw_network_tree> to account
    for the duplication of input layers with different filters. Each connection
    dictionary in network['connections'] of which the source layer has a
    'filters' entry (in the formatted layer flat dictionary) is replicated n
    times with different names.

    If the network level parameter 'scale_input_weights' is True, the newly
    created connections' weights are divided by n. Being lower in the tree,
    these updated weights will precede.

    Args:
        - <raw_network_tree>: Original tree defining the network (from the full
            params tree).

    Returns:
        - (dict): Dictionary similar to <raw_network_tree> except that the
            'connections' subdictionary has been expanded.

    """
    # Non expanded layers dict, used to read layer names and parameters before
    # layer name modifications/layer expansion
    layers = get_layers(raw_network_tree['layers'], expanded=False)

    raw_network_tree.update(
        {'connections': struct.flatten(
            [duplicate_connection(conn,
                                  raw_network_tree['connection_models'],
                                  layers)
             for conn in raw_network_tree['connections']])})

    return raw_network_tree


def duplicate_connection(conn, connection_models, layers_dict):
    """Possibly duplicate connections with different names.

    From a single connection dictionary, returns a list of duplicated
    connection dictionaries with different input layer names if the input layer
    should be expanded.
    """
    source = conn['source_layer']
    layer_params = layers_dict[source]['params']

    if 'filters' in layer_params.keys():
        exp_source_names = filt.get_expanded_names(source,
                                                   layer_params['filters'])
        if layer_params['scale_input_weights']:
            conn['params']['weights'] = (base_conn_weight(conn,
                                                          connection_models)
                                         / len(exp_source_names))
        return [
            struct.deepcopy_dict(conn, {'source_layer': exp_source_name})
            for exp_source_name in exp_source_names
        ]
    else:
        return [conn]


def base_conn_weight(conn, connection_models):
    """Return the base weight of the connection model <conn> inherits from."""
    return connection_models[conn['connection']]['weights']


def get_multimeter(pop_params):
    """Create a 'multimeter' dictionary from a population parameter dict.

    Args:
        - <pop_params> (dict): parameter dictionary of a single population.

    Return:
        (dict) : Dictionary of the form
            {'record_pop': <bool>,
             'rec_params': <rec_params>}
            where <bool> indicates whether the population should be recorder and
            <rec_params> is a nest-readable dictionary describing the
            multimeter connected to that population.

    """
    return {'record_pop': pop_params['record_multimeter'],
            'rec_params': {'record_from': pop_params['mm_record_from'],
                           'record_to': pop_params['mm_record_to'],
                           'interval': pop_params['mm_interval'],
                           'withtime': pop_params['mm_withtime'],
                           'withgid': pop_params['mm_withgid']}}


def get_spike_detector(pop_params):
    """Create a 'spike_detector' dictionary from a population parameter dict.

    Args:
        - <pop_params> (dict): parameter dictionary of a single population.

    Return:
        (dict) : Dictionary of the form
            {'record_pop': <bool>,
             'rec_params': <rec_params>}
            where <bool> indicates whether the population should be recorder and
            <rec_params> is a nest-readable dictionary describing the
            spike_detector connected to that population.

    """
    return {'record_pop': pop_params['record_spike_detector'],
            'rec_params': {'record_to': pop_params['sd_record_to'],
                           'withtime': pop_params['sd_withtime'],
                           'withgid': pop_params['sd_withgid']}}


def expand_populations(pop_list, non_expanded_layers):
    """Expand the list of populations to account for multiple filters.

    Args:
        - <pop_list>: list of tuples of the form (<pop_name>, <pop_params>) for
            population. <pop_params> contains eg the non-expanded layer name.
        - <non_expanded_layers>: non name-expanded formatted flat
            dictionary (from get_Network())
    Returns:
        (list): list of tuples duplicated with the same structure as pop_list
            but in which params['layer'] has been updated with the extended
            names.

    """
    expanded_list = []
    for (pop_name, pop_params) in pop_list:
        layer_name = pop_params['layer']
        layer_params = non_expanded_layers[layer_name]['params']
        if 'filters' in layer_params.keys():
            expanded_list += [(pop_name, ChainMap({'layer': ext_layer_name},
                                                  pop_params))
                              for ext_layer_name
                              in filt.get_expanded_names(
                                  layer_name,
                                  layer_params['filters'])]
        else:
            expanded_list += [(pop_name, pop_params)]

    return expanded_list


def get_populations(raw_network_tree):
    """Return nest-readable multimeters and spike detectors information.

    Args:
        - <raw_network_tree> (dict): Original tree defining the network (from
            the full params tree).

    Returns:
        - (list): List of dictionaries each of the form
                {'layer': <layer_name>,
                 'population': <pop_name>,
                 'mm': <multimeter_params>,
                 'sd': <spike_detectors_params>}
            where <multimeter_params> and <sd_params> are dictionaries of the
            form: {'record_pop': <bool>,
                   'rec_params': <nest_readable_dict>} describing the type of
            the multimeter or the spike_detector that would be connected to
            that specific population, and whether the population should be
            recorded.

    """
    pop_tree = raw_network_tree['populations']
    non_expanded_layers = get_layers(raw_network_tree['layers'], expanded=False)

    return [{'layer': pop_params['layer'],
             'population': pop_name,
             'mm': get_multimeter(pop_params),
             'sd': get_spike_detector(pop_params)}
            for (pop_name, pop_params)
            in expand_populations(struct.traverse(pop_tree,
                                                  params_key='params',
                                                  children_key='children',
                                                  name_key='name',
                                                  accumulator=[]),
                                  non_expanded_layers)]
