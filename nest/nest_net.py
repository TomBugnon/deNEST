import nest
from collections import ChainMap


def get_Network(archi, params):
    models = get_Models(archi[neurons], params)
    layers = get_Layers(archi, params)
    conns = get_Connections(archi, params)
    return models, layers, conns


def get_Models(archi):
        """ Returns a dictionary of which keys are the base nest models and
        values are lists of tuples  of the form (<model_name>,<params_chainmap>)
        where each tuple describes a model that will be used in the network. """

    return {
        linearize(archi['neurons']) + linearize(archi['synapses'])
    }

def linearize_Model_Tree(model_tree):
    """
    For a given tree input (eg neuron or synapse), returns a dictionary of which
    keys are the base nest models and values are lists of tuples  of the form
    (<name>,<params_chainmap>) where each tuple describes a model that will be used
    in the network. """

    return {
        base_name: [
            traverse_tree(base_tree, [], base_name
                          leaf_key='name',
                          params_key='params')
        ] for base_name, base_tree in model_tree.items()
    }


def traverse_Tree(
                  node, accumulator, current_node_name,
                  params_key='params',
                  children_key='children'):

    """Recursively traverses a tree and returns, for each leaf, its key and a
    ChainMap containing the ordered contents of the 'params_key' field (if
    existing) in each of the parent nodes.

    Args
        params_key (str): Lookup parameters with this key.
        accumulator (ChainMap): Append the newly accumulated parameters to
            this ChainMap. Used for recursion.
        current_node_name (str): key of the current node.
    Returns:
        list: list of tuples of the form (<leaf_name>, <params_chainmap>) where
            params_chainmap represents the ordered parameters collected in the
            parent nodes and leaf_name is the key under which each leaf is saved in
            the tree
    """

    # leaf
    if children_key not in node | node[children_key] is None:
        return [(current_node_name, ChainMap(node[params_key], accumulator))]
    # not leaf
    else:
        new_accumulator = ChainMap(node[params_key], accumulator)
        recursive_result = [
                            traverse_tree(
                                          node[children_key][name_key],
                                          new_accumulator,
                                          name_key,
                                          params_key=params_key,
                                          children_key=children_key)
                            for name_key in node[children_key]
                            ]
        # As the returned value for a node is a singleton list containing
        # one tuple, <recursive_result> is now a list of lists that we want
        # to flatten
        return [item for sublist in recursive_result for item in sublist]

def get_Layers(layers, params):
    """
    Returns list of tuples of the form (<layer_name>, <layer_params>) for each
    layer present in the network.
    Some general specifications of the network, such as layer size, ratio of
    excitatory and inhibitory neurons, presence or absence of edgewrapping, etc.
    are integrated from the <params> tree.
    Layer-specific parameters are derived from the <layers> tree."""

    all_layers = []
    input_area_name = params['input']['input_area_name']

    for layer_name in layers

        # The input layer handled separately, as it is duplicated as many times
        # as there are different filters
        if layers[layer][area] == input_area_name:
            all_layers += get_input_layers(layers[layer], layer_name, params)
        else:
            all_layers += get_area_layer(layers[layer], layer_name, params)
    return all_layers


def get_input_layers(specs, input_layer_name_base, params):

    input_layers = []

    # Define input layer names, depending on the different filters
    filters = params['input']['filters']
    input_layer_names = [input_layer_name_base + suffix
                         for suffix in get_input_suffixes(filters)]


def get_input_suffixes(filters):

    suffixes = []
    for dim in filters['dimensions']:
        if filters['dimensions']['dim'] > 1:
            dim_su
            suffixes = suffixes.append([filters['suffixes'][dim]
