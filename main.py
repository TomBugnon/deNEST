import os
import pprint

import yaml
from nestify.format_net import chaintree, get_Network
from nestify.init_nest import init_Network


def load_sim(full_sim_params):

    base_path = os.path.dirname(__file__)
    simtrees = [load_yaml(base_path, simtree_path)
                for simtree_path in load_yaml(full_sim_params)]
    return chaintree(simtrees)


def load_yaml(*args):
    f = open(os.path.join(*args), 'r')
    return yaml.load(f)


if __name__ == '__main__':

    # Which simulation should be run.
    full_sim_params = './sim_params/full_sim/default_sim.yaml'

    # load and merge all the simulation parameters
    full_sim = load_sim(full_sim_params)

    # Get relevant parts of the full simulation tree
    net = full_sim['children']['network']['children']
    sim = full_sim['children']['sim_params']['children']

    net = get_Network(net)
    net = init_Network(net, sim)
