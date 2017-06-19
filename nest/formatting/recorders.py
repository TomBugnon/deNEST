from collections import ChainMap

# import nest
import numpy as np

from net import get_expanded_names, traverse, updateDicts


def get_multimeters(pop_list):
    """ Returns a list of dictionaries:
        {'layer': <layer_name>,
         'population': <pop_name>,
         'rec_params': <rec_params>}
    for all the populations of which params['record_multimeter'], where
    <rec_params> is a nest-readable dictionary describing each multimeter.
    """
    return [
        {'layer': params['layer'],
         'population': pop_name,
         'rec_params': {'record_from': params['mm_record_from'],
                        'record_to': params['mm_record_to'],
                        'interval':params['mm_interval'],
                        'withtime': params['mm_withtime'],
                        'withgid': params['mm_withgid']}
         }
        for (pop_name, params) in pop_list
        if params['record_multimeter']
    ]


def get_spike_detectors(pop_list):
    """ Returns a list of dictionaries:
        {'layer': <layer_name>,
         'population': <pop_name>,
         'rec_params': <rec_params>}
    for all the populations of which params['record_multimeter'], where
    <rec_params> is a nest-readable dictionary describing each spike_detector.
    """
    return [
        {'layer': params['layer'],
         'population': pop_name,
         'rec_params': {'withtime': params['sd_withtime'],
                        'withgid': params['sd_withgid']}
         }
        for (pop_name, params) in pop_list
        if params['record_spike_detector']
    ]


def expand_populations(pop_list, non_expanded_layers):
    """ Duplicates with different names the tuples in pop_list if their layer
    has a 'filter' key.

    Args:
        - <pop_list>: list of tuples of the form (<pop_name>, <params>) for
            population. <params> contains the non-expanded layer name.
        - <non_expanded_layers>: non name-expanded formatted flat
            dictionary (from get_Network())
    Returns:
        (list) of tuples duplicated with the same structure as pop_list but in
            which params['layer'] has been updated with the extended names.
    """
    expanded_list = []
    for (pop_name, params) in pop_list:
        layer_name = params['layer']
        layer_params = non_expanded_layers[layer_name]['params']
        if 'filters' in layer_params.keys():
            expanded_list += [(pop_name, ChainMap({'layer': ext_layer_name},
                                                  params))
                              for ext_layer_name
                              in get_expanded_names(layer_name,
                                                    layer_params['filters'])]
        else:
            expanded_list += [(pop_name, params)]

    return expanded_list


def get_recorders(pop_tree, non_expanded_layers):
    """ Return nest-readable multimeters and spike detectors information.
    Args:
        - <pop_tree> (dict): Tree in which the recorders are defined. Leaves
            are populations.
        - <non_expanded_layers> (dict): Non name-expanded formatted flat
            dictionary (from get_Network()) used to expand the population tree.

    Returns:
        - (<multimeters>, <spike_detectors>) (tuple): <multimeters> and
            <spike_detectors> are both lists of dictionaries of the form:
                {'layer': <layer_name>,
                 'population': <pop_name>,
                 'rec_params': <rec_params>}
            where <rec_params> is a nest-readable dictionary describing either
            the multimeter or the spike_detector connected to that specific
            population.
            Only the recorders that we want to record from are returned in the
            list (cf 'record_multimeter' and 'record_spike_detector' keys in
            pop_tree)
    """

    pop_list = expand_populations(traverse(pop_tree,
                                           params_key='params',
                                           children_key='children',
                                           name_key='name',
                                           accumulator=[]),
                                  non_expanded_layers)

    return (get_multimeters(pop_list), get_spike_detectors(pop_list))
