import os
import pprint

import yaml
from net import chaintree, get_Network

if __name__ == '__main__':

    script_path = os.path.dirname(__file__)

    df = open(os.path.join(script_path,
                           '../../nets/default_visnet.yaml'), 'r')
    nf = open(os.path.join(script_path,
                           '../../nets/default_visnet_layer_params.yaml'), 'r')
    recs = open(os.path.join(script_path,
                             '../../nets/default_visnet_recorders.yaml'), 'r')
    df_nest = open(os.path.join(script_path,
                             '../../nets/default_nest_params.yaml'), 'r')

    default = yaml.load(df)
    params = yaml.load(nf)
    recorders = yaml.load(recs)
    nestp = yaml.load(df_nest)
    raw_net = chaintree([params, recorders, default, nestp])
    net = get_Network(raw_net['children'])

    from init_nest import *
    init_Kernel(raw_net['params'])
    create_Neurons(net['neuron_models'])
    create_Synapses(net['synapse_models'])
    create_Layers(net['layers'])
    create_Connections(net['connections'], net['layers'])
    connect_Recorders(net['populations'], net['layers'])
    print('Network fully initialized in NEST.')
