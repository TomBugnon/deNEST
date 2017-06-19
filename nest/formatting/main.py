import os
import pprint

import yaml
from net import chaintree, get_Network
from recorders import get_recorders

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
    print('network fetched')
    # pprint.pprint(net)

    recs = open(os.path.join(script_path,
                             '../../nets/default_visnet_recorders.yaml'), 'r')
    recorders = yaml.load(recs)
    # pprint.pprint(recorders)
    (mm, sd) = get_recorders(recorders['populations'],
                             net['non_expanded_layers'])
    print('Recorders fetched')
    # pprint.pprint(mm)
    # pprint.pprint(sd)
    print('success')
