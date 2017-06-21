from collections import ChainMap

import nest
import nest.topology as tp
import numpy as np


def initialize_network():
    return


def init_Kernel(nest_params):
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': nest_params['local_num_threads'],
                          'resolution': float(nest_params['resolution'])})

    msd = nest_params['seed']
    N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    pyrngs = [np.random.RandomState(s) for s in range(msd, msd + N_vp)]
    nest.SetKernelStatus({'grng_seed': msd + N_vp})
    nest.SetKernelStatus({'rng_seeds': range(msd + N_vp + 1,
                                             msd + 2 * N_vp + 1)})
    nest.SetStatus([0], {'print_time': nest_params['print_time']})
    nest.ResetNetwork()
    return


def create_Neurons(neuron_models):
    print('Create neurons...')
    for (base_nest_model, model_name, params_chainmap) in neuron_models:
        nest.CopyModel(base_nest_model, model_name, dict(params_chainmap))
    return
    print('Done.')


def create_Synapses(synapse_models):
    """ NEST expects an 'receptor_type' index rather than a 'receptor_type'
    string to create a synapse model. This index needs to be found through nest
    in the defaults of the target neuron.
    """
    print('Create synapses...')
    for (base_nest_model, model_name, params_chainmap) in synapse_models:
        nest.CopyModel(base_nest_model,
                       model_name,
                       dict(format_synapse_params(params_chainmap)))
    return
    print('Done.')

def format_synapse_params(syn_params):

    assert(not syn_params or len(syn_params.keys()) == 2)
    formatted = {}

    if 'receptor_type' in syn_params.keys():
        tgt_type = syn_params['target_neuron']
        receptors = nest.GetDefaults(tgt_type)['receptor_types']
        formatted['receptor_type'] = receptors[syn_params['receptor_type']]

    return formatted


def create_Layers(layers):
    """Create layers and record the nest gid of the layer under the 'gid' key
    of each layer's dictionary. Layers is a flat dictionary of dictionaries.
    """
    print('Create layers...')
    for layer_name, layer_dict in layers.items():
        gid = tp.CreateLayer(dict(layer_dict['nest_params']))
        layers[layer_name].update({'gid': gid})
    return
    print('Done')


def create_Connections(connections, layers):

    assert ('gid' in layers[list(layers)[0]]), 'Please create the layers first'
    print('Create connections...')
    for (source_layer, target_layer, conn_params) in connections:
        tp.ConnectLayers(layers[source_layer]['gid'],
                         layers[target_layer]['gid'],
                         dict(conn_params))
    print('Done.')
    return


def connect_Recorders(pop_list, layers):
    """ Connect the recorder and the populations and modify in place the list of
    population directory to save the recorders' nest gid.

    Args:
        - <pop_list> (list of dicts): Each dictionary is of the form:
                {'layer': <layer_name>,
                 'population': <pop_name>,
                 'mm': {'record_pop': <bool>,
                        'rec_params': {<nest_multimeter_params>},
                 'sd': {'record_pop': <bool>,
                        'rec_params': {<nest_multimeter_params>}}
     Return:
        - (list of dicts): modified list of dictionaries, updated with the key
          'gid' and its value for each recorder.
            eg:
                 'mm': {'gid': <value>,
                        'record_pop': <bool>,
                        'rec_params': {<nest_multimeter_params>},
            NB: if <bool>==False, 'gid' is set to False.
    """
    print('Connect recorders')
    [connect_rec(pop, recorder_type, layers)
     for pop in pop_list
     for recorder_type in ["multimeter", "spike_detector"]]
    print('Done.')
    return pop_list


def connect_rec(pop, recorder_type, layers):
    """ Possibly connect the multimeter or spike_detector on the population
    described by pop and modify in place the pop dictionary with the gid
    of the recorder or False if no recorder has been connected.
    """

    rec_key = ('mm' if recorder_type == "multimeter" else 'sd')

    if pop[rec_key]['record_pop']:
        # import ipdb; ipdb.set_trace()

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
    print('.', end='')
    return
