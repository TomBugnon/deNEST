#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/init_nest.py

"""Initialize NEST kernel and network."""

import itertools

import nest
import nest.topology as tp
import numpy as np
from tqdm import tqdm

from ..utils.system import mkdir_ifnot


def init_nest(network, kernel_params):
    """Initialize NEST kernel and network.

    Args:
        - network (Network): Modified in place with GIDs and unit positions in
            layer for each population.
        - kernel_params (dict): Kernel parameters (from full parameter tree).

    """
    print('Initializing kernel...')
    init_kernel(kernel_params)
    print('Initializing network...')
    nest.ResetNetwork()
    create_neurons(network['neuron_models'])
    create_synapses(network['synapse_models'])
    create_layers(network['layers'])
    create_connections(network['connections'], network['layers'])
    connect_recorders(network['populations'], network['layers'])
    print('Network has been successfully initialized.')


def init_kernel(kernel_params):
    """Initialize NEST kernel."""
    nest.ResetKernel()
    nest.SetKernelStatus(
        {'local_num_threads': kernel_params['local_num_threads'],
         'resolution': float(kernel_params['resolution'])})
    msd = kernel_params['seed']
    N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    pyrngs = [np.random.RandomState(s) for s in range(msd, msd + N_vp)]
    nest.SetKernelStatus({
        'grng_seed': msd + N_vp,
        'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
        'print_time': kernel_params['print_time'],
    })


def gid_location_mapping(layer_gid, population_name):
    """Create the mapping between population units' GID and layer location.

    Args:
        - <layer_gid> (tuple): Singleton tuple of int containing the GID of the
            considered layer.
        - <population_name> (str): Name of the considered layer's population.

    Returns:
        (dict): Dictionary of the form
                    {'gid': <gid_by_location_array>,
                     'location': <location_by_gid_mapping>}
            where:
                - <gid_by_location_array> (np-array) is a (nrows, ncols)-array
                    of the same dimension as the layer. It is an array of
                    lists (possibly singletons) as there can be multiple units
                    of that population at each location.
                - <location_by_gid_mapping> (dict) is dictionary of which keys
                    are GIDs (int) and entries are (row, col) location (tuple
                    of int)

    """
    # Get layer resolution
    layer_topo = nest.GetStatus(layer_gid, 'topology')[0]
    nrows, ncols = layer_topo['rows'], layer_topo['columns']

    # Initialize bi-directional mapping dictionary.
    gid_loc_map = {'gid': np.empty((nrows, ncols), dtype=list),
                   'location': {}}

    # Iterate on all locations of the grid-based layer.
    for (i, j) in itertools.product(range(nrows), range(ncols)):

        # Get list of GIDs of all population units at that location
        location_units = [
            nd for nd in tp.GetElement(layer_gid,
                                       locations=(i, j))
            if nest.GetStatus((nd,), 'model')[0] == population_name
        ]

        # Update array of gids
        gid_loc_map['gid'][i, j] = location_units
        # Update mapping of locations
        gid_loc_map['location'].update({gid: (i, j)
                                        for gid in location_units})
    return gid_loc_map


def set_nest_savedir(nest_tmp_savedir):
    """Tell kernel where to save raw recorder data.

    NEST will save the raw recorder activity in < simulation_savedir > /tmp

    Args:
        - < simulation_savedir > (str): Absolute path to the directory in which
            all simulation parameters and formatted recorder activity will be
            saved.

    """
    mkdir_ifnot(nest_tmp_savedir)
    nest.SetKernelStatus({"data_path": nest_tmp_savedir})


def create_neurons(neuron_models):
    """Create neuron models in NEST."""
    for (base_nest_model,
         model_name,
         params_chainmap) in tqdm(neuron_models,
                                  desc='Create neurons: '):
        nest.CopyModel(base_nest_model, model_name, dict(params_chainmap))
    return
    print('Done.')


def create_synapses(synapse_models):
    """Create synapse models in NEST.

    NB: NEST expects an 'receptor_type' index rather than a 'receptor_type'
    string to create a synapse model. This index needs to be found through nest
    in the defaults of the target neuron.

    """
    for (base_nest_model,
         model_name,
         params_chainmap) in tqdm(synapse_models,
                                  desc='Create synapses: '):
        nest.CopyModel(base_nest_model,
                       model_name,
                       dict(format_synapse_params(params_chainmap)))
    return
    print('Done.')


def format_synapse_params(syn_params):
    """Format synapse parameters in a NEST readable dictionary."""
    assert not syn_params or len(syn_params.keys()) == 2, \
        ("""If you define 'receptor_type' for a synapse, I also expect
        target_neuron""")
    formatted = {}

    if 'receptor_type' in syn_params.keys():

        tgt_type = syn_params['target_neuron']
        receptors = nest.GetDefaults(tgt_type)['receptor_types']
        formatted['receptor_type'] = receptors[syn_params['receptor_type']]

    return formatted


def create_layers(layers):
    """Create layers and record GIDs.

    The nest GID of each layer is saved under the 'gid' key
    of each layer's dictionary.

    Args:
        <layers > (dict): Flat dictionary of dictionaries.
    """
    for layer_name, layer_dict in tqdm(layers.items(),
                                       desc='Create layers: '):
        gid = tp.CreateLayer(dict(layer_dict['nest_params']))
        layers[layer_name].update({'gid': gid})
    return


def create_connections(connections, layers):
    """Create NEST connections."""
    assert ('gid' in layers[list(layers)[0]]), 'Please create the layers first'
    for (source_layer,
         target_layer,
         conn_params) in tqdm(connections,
                              desc='Create connections: '):
        tp.ConnectLayers(layers[source_layer]['gid'],
                         layers[target_layer]['gid'],
                         dict(conn_params))
    return


def connect_recorders(pop_list, layers):
    """Connect the recorder to the populations.

    Modifies in place the list of population directory to save the recorders'
    nest gid.

    Args:
        - < pop_list > (list): List of dict, each of the form:
                {'layer': < layer_name >,
                 'population': < pop_name >,
                 'mm': {'record_pop': < bool >,
                        'rec_params': { < nest_multimeter_params > },
                 'sd': {'record_pop': < bool >,
                        'rec_params': { < nest_multimeter_params > }}
     Return:
        - (list): modified list of dictionaries, updated with the key 'gid' and
            its value for each recorder.
            eg:
                 'mm': {'gid': < value >,
                        'record_pop': < bool >,
                        'rec_params': { < nest_multimeter_params > },
            NB: if < bool > == False, 'gid' is set to False.

    """
    for pop in tqdm(pop_list,
                    desc='Connect recorders: '):
        [connect_rec(pop, recorder_type, layers)
         for recorder_type in ["multimeter", "spike_detector"]]
    return pop_list


def connect_rec(pop, recorder_type, layers):
    """Possibly connect a recorder to a given population.

    Possibly connect the multimeter or spike_detector on the population
    described by pop and modify in place the pop dictionary with the gid
    of the recorder or False if no recorder has been connected.

    A single recorder is not connected to a layer or a population, but rather
    connected to all the units of a given population within a given layer.

    """
    rec_key = ('mm' if recorder_type == "multimeter" else 'sd')

    if pop[rec_key]['record_pop']:

        gid = nest.Create(recorder_type, params=pop[rec_key]['rec_params'])
        layer_gid = layers[pop['layer']]['gid']
        tgts = [nd for nd in nest.GetLeaves(layer_gid)[0]
                if nest.GetStatus([nd], 'model')[0] == pop['population']]

        if recorder_type == 'multimeter':
            nest.Connect(gid, tgts)
        elif recorder_type == 'spike_detector':
            nest.Connect(tgts, gid)

        pop[rec_key]['gid'] = gid
    else:
        pop[rec_key]['gid'] = False
    return
