#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/format_net.py

"""Convert network parameters to a NEST-readable representation."""

import copy as cp
from collections import ChainMap

from ..utils import filter_suffixes as filt
from ..utils import structures as struct


def get_network(params):
    """Transform Net parameters into a form suitable for passing to NEST.

    Returns:
        dict: Dictionary of the form
                {'neuron_models': <neuron_models>,
                'synapse_models': <synapse_models>,
                'layers': <layers>,
                'non_expanded_layers': <non_expanded_layers>,
                'connections': <connections>,
                'areas': <areas>,
                'populations': <populations>},
            where:
            - <neuron_models> is list of tuples each of the form:
                    (<base_nest_model>, <model_name>, <params_chainmap>)
            - <synapse_models> is list of tuples each of the form:
                    (<base_nest_model>, <model_name>, <params_chainmap>)
            - <layers> is a dictionary of the form
                    {<layer_name>: {'params': <params_chainmap>
                                    'nest_params': <nest_params_chainmap>}
                where
                 - 'params' contains all the parameters related to this layer,
                 - 'nest_params' contains the nest_formatted parameters
                    used to create the layer,
            - <non_expanded_layers> is similar to layers but without layer
                duplication for different filters,
            - <connections> is a list of dictionaries each of the form:
                    {'source_layer': <source_layer_name>,
                     'target_layer': <target_layer_name>,
                     'source_population': <source_population_name>,
                     'target_population': <target_population_name>,
                     'params': <connection_params>,
                     'nest_params': <nest_connection_params>}
                where
                - <nest_params> is the nest-readable dictionary describing the
                    connection,
                - <connection_params> contains all the parameters for the
                    connection.
            - <areas> is a dictionary of the form:
                    {<area_name>: <list_of_layers>},
                where
                - <list_of_layers> is the list of all layers of the network
                  within a given area
            - <populations> is a list of dictionaries each of the form:
                    {'layer': <layer_name>,
                     'population': <population_name>,
                     'mm': <multimeter_params>,
                     'sd': <spike_detectors_params>}

    """
    layers = get_layers(params['layers'], expanded=True)
    return {
        'neuron_models': get_models(params['neuron_models']),
        'synapse_models': get_models(params['synapse_models']),
        'layers': layers,
        'connections': get_connections(params),
        'areas': get_areas(layers),
        'populations': get_populations(params),
    }


def get_models(models):
    """Return the leaf models in a model dictionary.

    Args:
        models (Params): Parameter for models. Should contain at depth 0 under
        the key `nest_model` the name of the base nest model in the tree
        inherits of.

    Returns:
        list: List of tuples  of the form:
                (<base_nest_model, <model_name>, <params_chainmap)
            where each tuple describes a model that will be used in the
            network.

    """
    return struct.flatten([
        struct.distribute_to_tuple(
            struct.traverse(model),
            model['nest_model'],
            pos=0)
        for key, model in models.items()
    ])


def get_layers(layers_tree, expanded=True):
    """Generate a flat dictionary describing the layers.

    Args:
        layers_tree (dict): Tree that will be traversed to gather all
            parameters of each layer-leaf. The gathered parameters are then
            formatted to produce NEST-readable parameter dictionaries.
        expanded (bool): If True, returns the expanded tree after taking into
            account the replication of layers for different filters, otherwise
            don't replicate layers whatsoever.

    Returns:
        dict: Dictionary of the form:
            {<layer_name>: {'params': <params_chainmap>
                            'nest_params': <nest_params_chainmap>}
             where
             - 'params' contains all the parameters related to this layer,
             - 'nest_params' contains the nest_formatted parameters
                used to create the layer.

    """
    # List of tuples of the form (<layer_name>, <params_chainmap>).
    layer_list = save_base_name(struct.traverse(layers_tree))
    if expanded:
        # The layers whose <params_chainmap> contains the field 'filters' are
        # replicated with different names.
        return format_layer_list(expand_layer_list(layer_list))
    return format_layer_list(layer_list)


def save_base_name(layer_list):
    """Save input layer name before extension.

    Add the pre-extension layer name to the 'base_name' key of the
    parameters for each layer.

    Args:
        layer_list (list): List of tuples of the form
            (<layer_name>, <params_chainmap>).
    """
    return [(name, params_chainmap.new_child({'base_name': name}))
            for (name, params_chainmap) in layer_list]


def format_layer_list(layer_list):
    """Generate a layer dictionary from a layer list.

    Args:
        layer_list (list): List of the form
            [(<layer_name>, <params_chainmap>), ...]

    Returns:
        dict: Dictionary of the form
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
    return {
        'rows': layer_params['nrows'],
        'columns': layer_params['ncols'],
        'extent': [layer_params.get('visSize', float(layer_params['nrows'])),
                   layer_params.get('visSize', float(layer_params['ncols']))],
        'edge_wrap': layer_params.get('edge_wrap', True),
        'elements': get_layer_elements(layer_params),
    }


def get_layer_elements(layer_params):
    """Format for nest the list of elements of a layer.

    Returns:
        dict: Dictionary of the form:
                {'elements': <elements_list>}
            where <elements_list> is of the form:
                ['L23_exc', 2, 'L23_inh', 1, 'L4_exc' , 2, ...]

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

    Invert the layer dictionary by reading the ['params']['area'] subkey of
    each layer to create an area dictionary.
    """
    return struct.invert_dict(
        {layer: layer_params['params']
         for (layer, layer_params) in layer_dict.items()},
        inversion_key='area'
    )


def get_connections(network):
    """Return the formatted connections of the network.

    Args:
        network (Params): Network parameters.

    Returns:
        list: List of dictionaries each describing a single
            population-to-population connection and of the form:
                    {'source_layer': <source_layer_name>,
                     'target_layer': <target_layer_name>,
                     'source_population': <source_population_name>,
                     'target_population': <target_population_name>,
                     'params': <connection_params>,
                     'nest_params': <nest_connection_params>}
                where
                - <nest_params> is the nest-readable dictionary describing the
                    connection,
                - <connection_params> contains all the parameters for the
                    connection.
        list: list of tuples of the form:
                (<source_layer>, <target_layer>, <nest_connection>)
            where each tuple describes a connection after taking into account
            possible duplication of input layers.

    """
    # Make a dictionary out of the tree of connection models
    network['connection_models'] = {model_name: model_params
                                    for (model_name, model_params)
                                    in struct.traverse(network['connection_models'])}

    expanded_network = expand_connections(network)
    expanded_layers = get_layers(expanded_network['layers'], expanded=True)

    all_connections = []
    for connection in expanded_network['connections']:
        nest_params = get_connection_params(connection,
                                            network['connection_models'],
                                            expanded_layers)
        all_connections.append(
            {'source_layer': connection['source_layer'],
             'target_layer': connection['target_layer'],
             'source_population': connection['source_population'],
             'target_population': connection['target_population'],
             'nest_params': nest_params,
             'params': ChainMap(connection['params'], nest_params)}
        )
    return all_connections


def get_connection_params(connection, models, layers):
    """Return NEST-readable connection parameters.

    Args:
        connection (dict): Connection parameters.
        models (dict): Models of connections.
        layers (dict): Expanded and formatted (flat) layer dictionary.

    Return:
        ChainMap: Chainmap describing a given connection. The dictionary given
            to NEST to describe the connection is ``dict(ChainMap)``.

    """
    source_params = layers[connection['source_layer']]['params']
    target_params = layers[connection['target_layer']]['params']

    # Update connection models with the specific connection params (possibly
    # empty).
    params = ChainMap(connection['nest_params'],
                      models[connection['connection']])

    source_size = max(source_params['nrows'], source_params['ncols'])

    # TODO: RF and weight scaling:
    # - maskfactor?
    # - btw error in ht files: secondary horizontal intralaminar mixes dcpS and
    #   dcpP
    if not target_params.get('scale_kernels_masks', False):
        rf_factor = 1.
    else:
        # If we want to give the masks and kernel sizes in 'number of units', we
        # have to scale as NEST # expects a mask in 'extent' of the layer.
        rf_factor = (target_params.get('rf_scale_factor', 1.)
                     * source_params['visSize'] / (source_size - 1))
    return params.new_child(
        {'sources': {'model': connection['source_population']},
         'targets': {'model': connection['target_population']},
         'mask': scaled_mask(params.get('mask', {}), rf_factor),
         'kernel': scaled_kernel(params['kernel'], rf_factor),
         'weights': scaled_weights(params['weights'],
                                   source_params.get('weight_gain', 1))})


def scaled_mask(mask_dict, scale_factor):
    """Scale the size of a connection mask by `scale_factor`."""
    keys = list(mask_dict.keys())
    assert len(keys) <= 1, 'Wrong formatting of connection mask'
    mask_dict_copy = cp.deepcopy(mask_dict)

    if 'circular' in keys:
        mask_dict_copy['circular']['radius'] *= scale_factor
    elif 'rectangular' in keys:
        mask_dict_copy['rectangular'] = {
            key: [scale_factor * scalar for scalar in scalar_list]
            for key, scalar_list in mask_dict['rectangular'].items()
        }

    return mask_dict_copy


def scaled_kernel(kernel, scale_factor):
    """Scale the size of a connection kernel by `scale_factor`."""
    if isinstance(kernel, (float, int)):
        return float(kernel)
    elif isinstance(kernel, (dict)) and 'gaussian' in kernel.keys():
        kernel_copy = cp.deepcopy(kernel)
        kernel_copy['gaussian']['sigma'] *= scale_factor
    else:
        raise Exception('Wrong formatting of connection kernel')
    return kernel_copy


def scaled_weights(weights, scale_factor):
    """Multiply connection weights by `scale_factor."""
    if isinstance(weights, int):
        return weights * scale_factor
    Warning('Connection weight is not an integer: no scaling')
    return weights


def expand_layer_list(layer_list):
    """Duplicate with different names the layer tuples for different filters.

    Args:
        layer_list: List of tuples each of the form
            (<layer_name>, <params_chainmap>)

    Returns:
        list: List with the same format for which tuples have been duplicated
        with different <layer_name> if <params_chainmap> has a `filters` key.

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


def expand_connections(network):
    """Account for the duplication of input layers with different filters.

    Modifify the 'connections' subdictionary of <params> to account for the
    duplication of input layers with different filters. Each connection
    dictionary in network['connections'] of which the source layer has a
    'filters' entry (in the formatted layer flat dictionary) is replicated n
    times with different names.

    If the network level parameter 'scale_input_weights' is True, the newly
    created connections' weights are divided by n. Being lower in the tree,
    these updated weights will precede.

    Args:
        network (Params): The network parameters.

    Returns:
        Params: The network parameters with the 'connections' subtree expanded.

    """
    # Non expanded layers dict, used to read layer names and parameters before
    # layer name modifications/layer expansion
    layers = get_layers(network['layers'], expanded=False)

    network.update(
        {'connections': struct.flatten(
            [duplicate_connection(connection,
                                  network['connection_models'],
                                  layers)
             for connection in network['connections']])})

    return network


def duplicate_connection(connection, models, layers):
    """Possibly duplicate connections with different names.

    From a single connection dictionary, returns a list of duplicated
    connection dictionaries with different input layer names if the input layer
    should be expanded.
    """
    source = connection['source_layer']
    layer_params = layers[source]['params']

    if 'filters' in layer_params.keys():
        exp_source_names = filt.get_expanded_names(source,
                                                   layer_params['filters'])
        if layer_params.get('scale_input_weights', True):
            base_weight = base_conn_weight(connection, models)
            scaling_factor = 1./len(exp_source_names)
            connection['params']['weights'] = scaled_weights(base_weight,
                                                             scaling_factor)
        return [
            struct.deepcopy_dict(connection, {'source_layer': exp_source_name})
            for exp_source_name in exp_source_names
        ]
    else:
        return [connection]


def base_conn_weight(conn, connection_models):
    """Return the base weight of the connection model <conn> inherits from."""
    return connection_models[conn['connection']]['weights']


def get_multimeter(pop_params):
    """Create a 'multimeter' dictionary from population parameters.

    Args:
        pop_params (dict): Parameters for a single population.

    Return:
        dict: Dictionary of the form
                {'record_pop': <bool>,
                'rec_params': <rec_params>}
            where <bool> indicates whether the population should be recorder
            and <rec_params> is a nest-readable dictionary describing the
            multimeter connected to that population.

    """
    return {'record_pop': pop_params['record_multimeter'],
            'rec_params': {'record_from': pop_params.get('mm_record_from', 'V_m'),
                           'record_to': pop_params.get('mm_record_to', 'file'),
                           'interval': pop_params.get('mm_interval', 1.),
                           'withtime': pop_params.get('mm_withtime', True),
                           'withgid': pop_params.get('mm_withgid', True)}}


def get_spike_detector(pop_params):
    """Create a 'spike_detector' dictionary population parameters.

    Args:
        pop_params (dict): Parameters for a single population.

    Return:
        (dict) : Dictionary of the form
            {'record_pop': <bool>,
             'rec_params': <rec_params>}
            where <bool> indicates whether the population should be recorder and
            <rec_params> is a nest-readable dictionary describing the
            spike_detector connected to that population.

    """
    return {'record_pop': pop_params['record_spike_detector'],
            'rec_params': {'record_to': pop_params.get('sd_record_to', 'file'),
                           'withtime': pop_params.get('sd_withtime', True),
                           'withgid': pop_params.get('sd_withgid', True)}}


def expand_populations(pop_list, non_expanded_layers):
    """Expand the list of populations to account for multiple filters.

    Args:
        pop_list (list[tuple]): List of tuples of the form (<pop_name>, <pop_params>) for
            population. <pop_params> contains e.g. the non-expanded layer name.
        non_expanded_layers (dict): non name-expanded formatted flat dictionary
            (from ``get_network``)
    Returns:
        list: List of tuples duplicated with the same structure as pop_list but
        in which params['layer'] has been updated with the extended names.

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


def get_populations(network):
    """Return nest-readable multimeters and spike detectors information.

    Args:
        network (Params): Network parameters.

    Returns:
        list[dict]: List of dictionaries each of the form:
                {'layer': <layer_name>,
                 'population': <pop_name>,
                 'mm': <multimeter_params>,
                 'sd': <spike_detectors_params>},
            where <multimeter_params> and <sd_params> are dictionaries of the
            form:
                {'record_pop': <bool>,
                 'rec_params': <nest_readable_dict>},
            describing the type of the multimeter or the spike_detector that
            would be connected to that specific population, and whether the
            population should be recorded.

    """
    pop_tree = network['populations']
    non_expanded_layers = get_layers(network['layers'], expanded=False)

    return [{'layer': pop_params['layer'],
             'population': pop_name,
             'mm': get_multimeter(pop_params),
             'sd': get_spike_detector(pop_params)}
            for (pop_name, pop_params)
            in expand_populations(struct.traverse(pop_tree),
                                  non_expanded_layers)]
