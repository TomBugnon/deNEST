import itertools
import os
import pprint
from collections import ChainMap

# import nest
import numpy as np

import yaml


def get_Network(network):
    '''
    Returns:
        - dict:'neuron_models':<neuron_models>,
                 'synapse_models':<synapse_models>,
                 'layers':<layers>,
                 'connections':<connections>,
                 'areas':<areas>} where:
            - <neuron_models> is list of tuples each of the form:
                (<base_nest_model>,<model_name>,<params_chainmap>)
            - <synapse_models> is list of tuples each of the form:
                (<base_nest_model>,<model_name>,<params_chainmap>)
            - <layers> is a dictionary of the form:
                {<layer_name>: {'params': <params_chainmap>
                                'nest_params': <nest_params_chainmap>}
                 where
                 - 'params' contains all the parameters related to this layer,
                 - 'nest_params' contains the nest_formatted parameters
                    used to create the layer,
            - <connections> is a list of tuples each of the form:
                (<source_layer>, <target_layer>, <params_chainmap>)
            - <areas> is a dictionary of the form:
                {<area_name>: <list_of_layers>} where <list_of_layers> is the
                list of all layers of the network within a given area
    '''
    layers = get_Layers(network['layers'])
    return {
        'neuron_models': get_Models(network['neuron_models']),
        'synapse_models': get_Models(network['synapse_models']),
        'layers': layers,
        'connections': get_Connections(network),
        'areas': get_Areas(layers)
    }


def combine_networks(networks_list, children_key='children'):
    # "Chain all the trees in each network in network_list"
    # return {tree_name: chaintree(trees_list, children_key='children')
    #         for tree_name, trees_list in all_key_values(networks_list)}
    return


def get_Models(model_tree):
    """ Returns the leaf models in a model dictionary.

    Returns a lists of tuples  of the form: (<base_nest_model, <model_name>,
    <params_chainmap) where each tuple describes a model that will be used in
    the network. <model_tree> contains at depth 0 under the key 'nest_model' the
    name of the base nest model each model in the tree (leaf) inherits of.
    """
    return flatten([
        distribute_to_tuple(
            traverse(
                model_dict,
                params_key='params',
                children_key='children',
                name_key='name',
                accumulator=[]),
            model_dict['nest_model'],
            position=0) for key, model_dict in model_tree.items()
    ])


def get_Layers(layers_tree):
    """ Generates from a tree a flat dictionnary describing the
    layers-leaf of <layers_tree>

    Args:
        - <layers_tree> (dict): Tree that will be traversed to gather all
            parameters of each layer-leaf. The thus gathered parameters are
            then formatted to produce nest-readable parameter dictionaries.
    Returns:
        - dictionary of the form:
            {<layer_name>: {'params': <params_chainmap>
                            'nest_params': <nest_params_chainmap>}
             where
             - 'params' contains all the parameters related to this layer,
             - 'nest_params' contains the nest_formatted parameters
                used to create the layer,
    """
    # List of tuples of the form (<layer_name>, <params_chainmap>).
    # The layers whose <params_chainmap> contains the field 'filters' are
    # replicated with different names.
    layer_list = expand_layer_list(
        traverse(
            layers_tree,
            params_key='params',
            children_key='children',
            name_key='name',
            accumulator=[]))
    return format_layer_list(layer_list)


def format_layer_list(layer_list):
    """ Generates a dictionary of the form:
    {<layer_name>: {'params': <params_chainmap>,
                    'nest_params': <nest_params_chainmap>} from a layer_list of
    the form (<layer_name>, <params_chainmap>)
    """

    return {
        layer_name: {
            'params': params,
            'nest_params': format_nest_layer_params(params)
        }
        for (layer_name, params) in layer_list
    }


def format_nest_layer_params(layer_params):

    nest_p = {}
    nest_p['rows'] = layer_params['size']
    nest_p['columns'] = layer_params['size']
    nest_p['extent'] = layer_params['visSize']
    nest_p['edge_wrap'] = layer_params['edge_wrap']
    nest_p['elements'] = get_layer_elements(layer_params)

    return nest_p


def get_layer_elements(layer_params):
    """ Returns:
    {'elements': <elements_list>} where <elements_list> is eg of the form:
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


def get_Areas(layer_dict):
    """Create an area dictionary from the layer dictionary.

    Invert the layer dictionary by reading the 'params':'area' subkey of each
    layer to create an area dictionary.
    """
    return invert_dict({layer: layer_params['params']
                        for (layer, layer_params) in layer_dict.items()},
                       inversion_key='area')


def get_Connections(network):
    """Return connections.

    Returns a list of tuples each of the form: (<source_layer>, <target_layer>,
    <params_chainmap>) where each tuple describes a connection after taking in
    account possible duplication of input layers.
    """
    network = expand_connections(network)
    layers = get_Layers(network['layers'])

    return [(conn['source_layer'], conn['target_layer'], get_conn_params(
        conn, network['connection_models'], layers))
        for conn in network['connections']]


def get_conn_params(conn, connection_models, layers):

    source_params = layers[conn['source_layer']]['params']
    target_params = layers[conn['target_layer']]['params']

    # Possibly update connection models
    if conn['params']:
        p = ChainMap(conn['params'], connection_models[conn['connection']])
    else:
        p = connection_models[conn['connection']]

    # RF scaling
    # TODO: maskfactor?
    # TODO: btw error in ht files: secondary horizontal intralaminar mixes dcpS
    # and dcpP
    rf_factor = (target_params['rf_scale_factor'] * source_params['visSize'] /
                 (source_params['size'] - 1))
    p['mask'] = scale_conn_mask(p['mask'], rf_factor)
    p['kernel'] = scale_conn_kernel(p['kernel'], rf_factor)
    # Weight scaling
    p['weights'] = scale_conn_weights(p['weights'],
                                      source_params['weight_gain'])
    p['sources'] = {'population': conn['source_population']}
    p['targets'] = {'population': conn['target_population']}

    return p


def scale_conn_mask(mask_dict, scale_factor):

    keys = list(mask_dict.keys())
    assert len(keys) <= 1, 'Wrong formatting of connection mask'

    if keys[0] == 'circular':
        mask_dict['circular']['radius'] *= scale_factor
    elif keys[0] == 'rectangular':
        mask_dict['rectangular'] = {
            key: [scale_factor * scalar for scalar in scalar_list]
            for key, scalar_list in mask_dict['rectangular'].items()
        }

    return mask_dict


def scale_conn_kernel(kernel, scale_factor):
    if isinstance(kernel, (float, int)):
        pass
    elif isinstance(kernel, (dict)) and 'gaussian' in kernel.keys():
        kernel['gaussian']['sigma'] *= scale_factor
    else:
        raise Exception('Wrong formatting of connection kernel')
    return kernel


def scale_conn_weights(weights, scale_factor):
    return (weights * scale_factor)


def get_area_layer_params(area, params):
    p = {}

    p['rows'] = params['resolution']['size']['area']
    p['columns'] = params['resolution']['size']['area']
    p['extent'] = params['resolution']['visSize']
    p['edge_wrap'] = params['resolution']['edge_wrap']

    return p


def check_connectivity():
    '''
    - input layers are only sources and never targets'''
    return


def expand_layer_list(layer_list):
    ''' Duplicates with different names the (<layer_name>, <params_chainmap>)
    tuples in layer_list if <params_chainmap> has the key expansion_key.
    Returns a list of tuples (similar to <layer_list>)
    '''

    expanded_list = []
    for (layer_name, params_chainmap) in layer_list:
        if 'filters' in params_chainmap.keys():
            expanded_list += [(ext_layer_name, params_chainmap)
                              for ext_layer_name in get_extended_names(
                                  layer_name, params_chainmap['filters'])]
        else:
            expanded_list += [(layer_name, params_chainmap)]

    return expanded_list


def expand_connections(network):
    """ Modifies the 'connections' subdictionary of the <network> dictionary to
    account for the duplication of input layers with different filters. Each
    connection dictionary in network['connections'] of which the source layer
    has a 'filters' entry (in the formatted layer flat dictionary) is replicated
    n times with different names. If the network level parameter
    'scale_input_weights' is True, the newly created connections' weights are
    divided by n.
    The output is a dictionary similar to network except that the 'connections'
    subdictionary has been extended.
    """
    # Non expanded layers dict, used to read layer names and parameters before
    # layer name modifications/layer expansion
    layers = format_layer_list(
        traverse(
            network['layers'],
            params_key='params',
            children_key='children',
            name_key='name',
            accumulator=[]))

    network.update(
        {'connections': flatten(
            [duplicate_connection(conn,
                                  network['connection_models'],
                                  layers)
             for conn in network['connections']])})

    return network


def duplicate_connection(conn, connection_models, layers_dict):
    """ From a single connection dictionary, returns a list of duplicated
    connection dictionaries with different input layer names if the input layer
    should be extended.
    """

    source = conn['source_layer']
    layer_params = layers_dict[source]['params']

    if 'filters' in layer_params.keys():
        exp_source_names = get_extended_names(source, layer_params['filters'])
        if layer_params['scale_input_weights']:
            conn['params']['weights'] = (base_conn_weight(conn, network)
                                         / len(exp_source_names))
        return [
            copy_dict(conn, {'source_layer': exp_source_name})
            for exp_source_name in exp_source_names
        ]
    else:
        return [conn]


def base_conn_weight(conn, network):
    """ Returns the base weight of the connection model <conn> derives from"""
    return network['connection_models'][conn['connection']]['weights']


def copy_dict(source_dict, diffs):
    """Returns a copy of source_dict, updated with the new key-value
       pairs in diffs."""
    result = dict(source_dict)  # Shallow copy !
    result.update(diffs)
    return result


def get_extended_names(base_layer_name, filters):
    '''
    Args:
        - <base_layer_name> (str)
        - <filters> (dict)
    '''

    return [
        base_layer_name + suffix for suffix in get_extension_suffixes(filters)
    ]


def get_extension_suffixes(filters):
    """Returns a list of suffixes that will be appended to the input layer names
    to describe which combination of filter dimensions each layer corresponds
    to.
    - All potential suffixes start with an underscore.
    - Degenerate dimensions are omitted.
    - If there is only one filter type overall, returns a singleton list
      containing an empty string.
    - If there is eg: 2 spatial frequencies, 2 orientations, 1 sign, returns
    - ['_sf1o1', '_sf1o2', '_sf2o1', '_sf2o2']
    """
    suffixes = []
    for dim in [d for d in filters['dimensions']
                if filters['dimensions'][d] > 1]:
        suffixes.append([filters['suffixes'][dim] + str(i + 1)
                         for i in range(filters['dimensions'][dim])])
    return (['_' + s for s in combine_strings(suffixes)]
            if suffixes else '')


def combine_strings(s):
    """ Returns a list of the combination of strings in s.

    Args:
        - <s>: List of lists of strings, eg [['a1','a2'], ['b1','b2','b3'],...]
    Returns:
        <list>: List of the combinations (sets) of strings from the different
            sublists, with one string from each sublist, eg
                ['a1b1', 'a1b2', 'a1b3', 'a2b1',...]
    """
    return [''.join(tup) for tup in list(itertools.product(*s))]


def chaintree(tree_list, children_key='children'):
    """ Recursively combine the trees in <tree_list> using ChainMaps.

    If there is only one tree in tree_list, return it.
    Otherwise, return a tree of which each node with path <path> has:
    - for keys the set of all keys of the node at <path> in the tree_list trees
    - for value under <key> either:
        if <key> is not children_key:
            - the ChainMap of values under <key> at <path> of all the trees if
                all there is more than one value and they are all dictionaries
            - the value under <key> at <path> of the first tree in treelist for
                which this <key> at <path> exists if one of the values is not a
                dictionary, or if there is only one value.
        if <key> is children_key:
            - the recursively chained subtrees.

    NB: This function should be applied on eg params['layers'] directly or
        modified to act recursively on multiple keys (eg 'layers' and
        'children')
    """
    chained_tree = {}

    if len(tree_list) == 1:
        return tree_list[0]

    # Combine horizontally the values in all trees under each key except
    # <children_key>
    key_value_tups = all_key_values(tree_list, omit_keys=[children_key])
    chained_tree.update({key: combine_values(values)
                         for (key, values) in key_value_tups})

    # Recursively combine children if there are any.
    # <children> is a list of (<child_key>, <list_of_child_subtrees>
    children_tups = all_key_values([tree[children_key]
                                    for tree in tree_list
                                    if (children_key in tree
                                        and tree[children_key])])
    if len(children_tups) > 0:
        combined_subtrees = {child_key: chaintree(child_subtrees_list,
                                                  children_key='children')
                             for (child_key, child_subtrees_list)
                             in children_tups}
        chained_tree[children_key] = combined_subtrees

    return chained_tree


def combine_values(values):
    """ Return either the ChainMap of the list <values> if all its elements are
    mappings, or the first element of the list.
    """
    if len(values) > 1 and all([isinstance(x, dict) for x in values]):
        return ChainMap(*values)
    else:
        return(values[0])


def all_key_values(dict_list, omit_keys=[]):
    """ Return a list of tuples (<key>, <value_list>) for each unique key in
    the dictionaries of dict_list, omitting the keys in <omit_keys>.
    <value_list> is the list of values under a given key in each of the
    dictionaries, scanned from left to right.
    """
    all_keys = (set(flatten([list(d.keys())
                             for d in dict_list]))
                - set(omit_keys))
    return [(key, [d[key] for d in dict_list if key in d])
            for key in all_keys]


def invert_dict(d, inversion_key):
    """ Inverts a dictionary of the form {<name>:{<inversion_key>:<value>, ...}}
    into a dictionary of the form: {<value>:<name_list>} where <name_list> is
    a list of the keys <name> of which <value> is the inversion_key value.

    Example:
        a = {'key1':{'inversion_key': value1,
                     'other_key': x1},
             'key2':{'inversion_key': value1,
                          'other_key': x2}
             'key3':{'inversion_key': value2,
                          'other_key': x1}
        invert_dict(a) = {'value1':['key1', 'key2'],
                          'value2':[key3]}
    """
    return {value: [key for (key, entry) in d.items()
                    if entry[inversion_key] == value]
            for value in set([entry[inversion_key]
                              for _, entry in d.items()])}


def distribute_to_tuple(tuple_list, value, position=0):
    """ Inserts <value> at the position <position> of each tuple in <tuple_list>
    """
    return [tuple(insert_in_list(list(tup), value, position))
            for tup in tuple_list]


def insert_in_list(l, value, position=0):
    """ Insert value at specific position of list and returns updated list."""
    l.insert(position, value)
    return l


def flatten(l):
    """Flatten a list of lists.

    If <l> is a list of lists, return the list of items in the sublists.
    If some elements of <l> are not lists, don't iterate on them. Therefore
    flatten(l) == l if l is eg a list of tuples.
    """
    gen = (x if isinstance(x, list) else [x] for x in l)
    return [item for sublist in gen for item in sublist]


def traverse(tree, params_key='params', children_key='children',
             name_key='name', accumulator=[]):
    """Recursively traverses a tree and returns, for each leaf, the value of its
    <name_key> key and a ChainMap containing the ordered contents of the
    'params_key' field (if existing) in each of the parent nodes.

    Args:
        params_key (str): Lookup parameters with this key.
        accumulator (ChainMap): Append the newly accumulated parameters to
            this ChainMap. Used for recursion.
        children_key (str): Children of a node are list of trees under this key
        name_key (str): When reaching a leaf, its name is under the key
            name_key.
    Returns:
        list: list of tuples of the form (<leaf_name>, <params_chainmap>) where
            params_chainmap represents the ordered parameters collected in the
            parent nodes and leaf_name is the value of the 'name' key of each
            leaf
    """
    acc = list(accumulator)  # Avoid accumulation on parallel paths

    # Get the current params if the key exists and its value is not None
    if params_key in tree and tree[params_key]:
        acc.append(tree[params_key])
    # Base case: leaf
    if not tree.get(children_key, False):
        return (tree['name'], ChainMap(*acc[::-1]))
    # Recursive case: not leaf.
    return flatten([traverse(child,
                             params_key=params_key,
                             children_key=children_key,
                             name_key=name_key,
                             accumulator=acc)
                    for name, child in tree[children_key].items()])


if __name__ == '__main__':

    script_path = os.path.dirname(__file__)

    df = open(os.path.join(script_path,
                           '../../nets/default_visnet.yaml'), 'r')
    nf = open(os.path.join(script_path,
                           '../../nets/default_visnet_modifs.yaml'), 'r')
    default = yaml.load(df)
    params = yaml.load(nf)

    network = default
    network['layers'] = chaintree([params['layers'], default['layers']])
    net = get_Network(network)
    pprint.pprint(net)
    print('success')
