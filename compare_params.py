# pylint: disable-all
# flake8: noqa

import itertools
import pickle
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

from spiking_visnet.network import Network
from spiking_visnet.parameters import load_params

with open('ht2005_leo_keiko.pkl', 'rb') as f:
    o = pickle.load(f)
old_neurons, old_layers, old_connections = o

def name_map_2(l):
    return {x[1]: x for x in l}

def name_map_1(l):
    return {x[0]: x for x in l}


old_neurons = name_map_2(old_neurons)
old_layers = name_map_1(old_layers)


def new_to_old_conn(new):
    source = new['source_layer']
    target = new['target_layer']
    params = dict(new['nest_params'])
    return [source, target, params]


layermap = {
    'Vp': ['L23', 'L4', 'L56'],
    'Vs': ['L23', 'L4', 'L56']
}


def new_model(layer, modelname):
    for layertype, modeltypes in layermap.items():
        if layertype in layer:
            for m in modeltypes:
                modelname = modelname.replace(m, layertype)
    return modelname


def old_to_new_conn(old):
    d = old[2]
    dct = deepcopy(d)
    source = old[0]

    modeltypes = set(itertools.chain.from_iterable(layermap.values()))

    if 'sources' in d:
        source_pop = d['sources']['model']

        s = source_pop.split('_')[0]
        if s in modeltypes:
            source = source.split('_')
            source = '_'.join(source[:1] + [s] + source[1:])

        dct['sources']['model'] = new_model(source, source_pop)

    target = old[1]

    if 'targets' in d:
        target_pop = d['targets']['model']

        t = target_pop.split('_')[0]
        if t in modeltypes:
            target = target.split('_')
            target = '_'.join(target[:1] + [t] + target[1:])

        dct['targets']['model'] = new_model(target, target_pop)

    return (source, target, dct)


def new_to_old_neur(new):
    base_model = new[0]
    model = new[1]
    params = dict(new[2])
    return (base_model, model, params)


def new_to_old_layer(name, layer):
    params = dict(layer['nest_params'])
    return (name, params)


def new_to_old_syn(new):
    return new


def diff(d1, d2, name1='', name2='', indent=2, quiet=False, onlymissing=False,
         returnstr=False):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())
    missing2 = keys1 - keys2
    missing1 = keys2 - keys1
    just = max(len(k) for k in keys1 | keys2)
    msg = []
    for key in missing1:
        msg.append('- first is missing key: `{}`'.format(key))
    for key in missing2:
        msg.append('- second is missing key: `{}`'.format(key))
    both = keys1 & keys2
    diffs = []
    if both and not onlymissing:
        just = max(len(k) for k in keys1 & keys2)
        for key in both:
            v1, v2 = d1[key], d2[key]
            if v1 != v2:
                if isinstance(v1, dict) and isinstance(v2, dict):
                    msg.append('- different value for: `{}`:'.format(key))
                    msg += diff(v1, v2, indent=indent,
                                returnstr=True)
                else:
                    msg.append('- different value for: `{}`: {} vs {}'.format(key, v1, v2))
                diffs.append((key, v1, v2))

    if not both:
        msg.append('\nfirst and second share no keys!')
    if msg:
        if name1:
            msg.insert(0, 'differences between {} and {}:'.format(name1, name2))
        msg = [' '*indent + line for line in msg]
        if returnstr:
            return msg
        if not quiet:
            print('\n'.join(msg))
        if both:
            return diffs
        return 1
    return 0


overrides = {
    'children': {
        'simulation': {'param_file_path':
                       'networks/ht2005_thalamus+primary/ht2005_params.yml',
                       'user_savedir':
                       'networks/ht2005_thalamus+primary/output'},
        'sessions': {'params': {'user_input': None}}
    }
}
params = load_params('networks/ht2005_thalamus+primary+secondary/ht2005_params.yml',
                     overrides=overrides)

network_params = params['children']['network']['children']
kernel_params = params['children']['kernel']
sim_params = params['children']['simulation']

network = Network(network_params, sim_params)

neurons = name_map_2([new_to_old_neur(n) for n in network['neuron_models']])

synapses = name_map_2([new_to_old_syn(s) for s in network['synapse_models']])

layers = name_map_1([new_to_old_layer(name, layer) for name, layer in network['layers'].items()])

connections = [new_to_old_conn(c) for c in network['connections']]

### CHECK NEURONS ###
print('\n\nCHECKING NEURON MODELS')
print('-'*80 + '\n')

print('new neurons:')
print(sorted(neurons.keys()))
print('old neurons:')
print(sorted(old_neurons.keys()))
print('')

# Check all old exc are the same:
old_exc = [(name, n) for name, n in old_neurons.items() if 'exc' in name and 'L' in name]
for name1, n1 in old_exc:
    for name2, n2 in old_exc:
        diff(n1[2], n2[2])
# Check all old inh are the same:
old_inh = [(name, n) for name, n in old_neurons.items() if 'inh' in name and 'L' in name]
if all([[diff(n1[2], n2[2]) for name1, n1 in old_exc]
        for name2, n2 in old_exc]):
    print('All old L*_exc are the same')
if all([[diff(n1[2], n2[2]) for name1, n1 in old_inh]
        for name2, n2 in old_inh]):
    print('All old L*_inh are the same')
print("")


def diff_neur(new_name, old_name, **kwargs):
    return diff(neurons[new_name][2], old_neurons[old_name][2],
                name1=new_name, name2=old_name, **kwargs)

print('Check new vs. old neurons:\n')
# Check new exc same as old exc:
diff_neur('Vp_exc', 'L4_exc', indent=2)
# Check new inh same as old 2exc:
diff_neur('Vp_inh', 'L4_inh', indent=2)

diff_neur('Tp_exc', 'Tp_exc', indent=2)
diff_neur('Tp_inh', 'Tp_inh', indent=2)
diff_neur('Rp', 'Rp', indent=2)


### CHECK LAYERS ###

print('\n\nCHECKING LAYERS')
print('-'*80 + '\n')

print('new layers:')
print(sorted(layers.keys()))
print('old layers:')
print(sorted(old_layers.keys()))
print('')

def diff_layer(new_name, old_name, **kwargs):
    return diff(layers[new_name][1], old_layers[old_name][1],
                name1=new_name, name2=old_name, **kwargs)

print('Check new vs. old layers with same name:\n')
for name in set(layers.keys()) & set(old_layers.keys()):
    diff_layer(name, name)


### CHECK CONNECTIONS ###

print('\n\nCHECKING CONNECTIONS')
print('-'*80 + '\n')


def get_conn_key(conn):
    source_layer, target_layer = conn[0], conn[1]
    conn = conn[2]
    source_pop = conn.get('sources', dict()).get('model')
    target_pop = conn.get('target', dict()).get('model')
    synapse_model = conn.get('synapse_model')
    conn_type = conn['connection_type']
    return (source_layer, target_layer, source_pop, target_pop, synapse_model,
            conn_type)


# filtered_layers = ['Vs_horizontal', 'Vs_vertical',
#                    'Vs_cross', 'Ts_layer', 'Rs_layer']
filtered_layers = []
print('Filter out layers from old connections: ', ', '.join(filtered_layers))
old_connections = [conn for conn in old_connections
                   if conn[0] not in filtered_layers
                   and conn[1] not in filtered_layers]
print('Convert old connections to new layer setup')
old_connections = [old_to_new_conn(c) for c in old_connections]

print('number of new connections:')
print(len(connections), '(One of those is a parrot connection)')
print('number of old connections:')
print(len(old_connections))
print('')

cmap = defaultdict(list)
for c in connections:
    cmap[get_conn_key(c)].append(c[2])
# Sort connection objects
for k, v in cmap.items():
    cmap[k] = sorted(v, key=lambda x: sorted(x.keys()))

old_cmap = defaultdict(list)
for c in old_connections:
    old_cmap[get_conn_key(c)].append(c[2])
# Sort connection objects
for k, v in old_cmap.items():
    old_cmap[k] = sorted(v, key=lambda x: sorted(x.keys()))

print('Check new vs. old connections:\n')
print('  Keys are of the form:')
print('  \t(<source_layer>, <target_layer>, <source_population>, <target_population>, <synapse_type>, <connection_type>)')
print('')

def listdiff(l1, l2, indent=2):
    lendiff = len(l1) - len(l2)
    longer = 'FIRST' if lendiff > 0 else 'SECOND'
    shorter  = 'SECOND' if lendiff > 0 else 'FIRST'
    if lendiff != 0:
        print(' '*indent + '{} HAS {} MORE ITEMS THAN {}'.format(longer, lendiff, shorter))

    for i, (v1, v2) in enumerate(zip(l1, l2)):
        diff(v1, v2, name1='index {}'.format(i), name2='index {}'.format(i), indent=indent+2)
        print('')

diff(cmap, old_cmap, onlymissing=True)
print('\n')
d = diff(cmap, old_cmap, quiet=True)
for k, v1, v2 in d:
    print('  - {}'.format(k))
    print('')
    listdiff(v1, v2, indent=2)
    print('')
