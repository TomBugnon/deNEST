import os
import pprint

import yaml

from .misc_helpers.structures import chaintree
from .nestify.format_net import get_Network
from .nestify.init_nest import init_Network


def load_sim(full_sim_params):

    base_path = os.path.dirname(__file__)
    simtrees = [load_yaml(base_path, simtree_path)
                for simtree_path in load_yaml(base_path, full_sim_params)]
    return chaintree(simtrees)


def load_yaml(*args):
    with open(os.path.join(*args), 'r') as f:
        return yaml.load(f)


def run():

    # Which simulation should be run.
    full_sim_file = './sim_params/full_sim/default_sim.yaml'

    # load and merge all the simulation parameters
    full_sim_params = load_sim(full_sim_file)

    # Get relevant parts of the full simulation tree
    net_tree = full_sim_params['children']['network']['children']
    kernel_params = full_sim_params['children']['kernel']

    net = get_Network(net_tree)
    (net, kernel_init) = init_Network(net, kernel_params)
    return


if __name__ == '__main__':
    run()
