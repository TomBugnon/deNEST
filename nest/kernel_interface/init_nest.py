import nest


def init_nest(nest_params):
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': nest_params['local_num_threads'],
                          'resolution': nest_params['resolution'],
                          'grng_seed': nest_params['grng_seed'],
                          'print_time': nest_params['print_time']})
    nest.ResetNetwork()
    return


def create_Models(models):

    for (base_nest_model, model_name, params_chainmap) in models:
        nest.copyModel(base_nest_model, model_name, params_chainmap)
    return


def create_layers(layers):
    """Create layers and record the nest gid of the layer under the 'gid' key
    of each layer's dictionary. Layers is a flat dictionary of dictionaries.
    """

    for layer_name, layer_dict in layers.items():
        gid = tp.CreateLayer(layer_dict['nest_params'])
        layers[layer_name].update({'gid': gid})
    return layers


def create_Connections(connections, layers):

    for (source_layer, target_layer, conn_params) in connections:
        tp.ConnectLayers(layers[source_layer]['gid'],
                         layers[target_layer]['gid'],
                         conn_params)


def connect_recorder(recorder_list, recorder_type):
    """ Connect the recorder and the populations and save their nest gids.

    Args:
        - <recorder_list> (list): list of dictionaries of the form:
            {'layer': <layer_name>,
             'population': <pop_name>,
             'rec_params': <rec_params>}
        - <recorder_type>: 'multimeter' or 'spike_detector'
     Return:
        - (list): equivalent list of dictionaries, updated with the key 'gid'
            and its value for each multimeter.
    """

    assert(recorder_type in ['multimeter', 'spike_detector'])

    for rec in recorder_list:
        rec['gid'] = nest.Create(recorder_type, params=rec['rec_params'])
        tgts = [nd for nd in nest.GetLeaves(rec['layer'])
                if nest.GetStatus([nd], 'model')[0] == rec['population']]

        if recorder_type == 'multimeter':
            nest.Connect(rec['gid'], tgts)
        elif recorder_type == 'spike_detector':
            nest.Connect(tgts, rec['gid'])

    return recorder_list
