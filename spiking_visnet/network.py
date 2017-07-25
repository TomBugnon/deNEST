#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""Provides the `Network` class."""

import collections

from tqdm import tqdm

from .nestify.format_net import get_network
from .nestify.init_nest import (gid_location_mapping, init_nest,
                                set_nest_savedir)
from .save import generate_save_subdir_str, get_NEST_tmp_savedir

# TODO: move functionality from nestify/format_net to this class


class Network(collections.UserDict):
    """Network class."""

    def __init__(self, network_params):
        """Initialize object from parameter tree."""
        super().__init__(get_network(network_params))

    def get_save_subdir_str(self, full_params_tree, param_file_path):
        """Record the subdirectory string for this network/simulation."""
        self.save_subdir_str = generate_save_subdir_str(full_params_tree,
                                                        param_file_path)

    def init_nest(self, kernel_params):
        """Initialize NEST kernel and network.

        The GIDs of the created NEST objects are added in place to the Network
        object.
        Provides the NEST kernel with the absolute path to the temporary
        directory where it will save the recorders.
        """
        init_nest(self, kernel_params)
        # Tell NEST kernel where to save all the recorders.
        tmp_save_dir = join(SAVE_DIR, self.save_subdir_str, 'tmp')
        set_nest_savedir(tmp_save_dir)

    def input_layer(self):
        """Return (name, nest_params, params) for an arbitrary input layer."""
        name = self['areas']['input_area'][0]
        return (name,
                self['layers'][name]['nest_params'],
                self['layers'][name]['params'])

    def filters(self):
        """Return the 'filters' dictionary for an arbitrary input layer.

        If no filter is applied to the input layers, return -1
        """
        (_, _, params) = self.input_layer()
        return params.get('filters', -1)

    def input_res(self):
        """Return the resolution of the network's input layers."""
        (_, nest_params, _) = self.input_layer()
        return (nest_params['columns'], nest_params['rows'])

    def get_gid_location_mappings(self):
        """Create a self.location attribute with the location/GID mappings.

        Creates a self.location attribute containing a tree such that:
            self.gid_coo[<layer_name>][<population_name>] = <gid_coo_dict>
        where:
            <gid_coo_dict> is a dictionary of the form:
                    {'gid': <gid_by_location_array>,
                     'location': <location_by_gid_mapping>}
        Finally:
        - <gid_by_location_array> (np-array) is a (nrows, ncols)-array of the
            same dimension as the layer. It is an array of lists (possibly
            singletons) as there can be multiple units of that population at
            each location.
        - <location_by_gid_mapping> (dict) is dictionary of which keys are
            GIDs (int) and entries are (row, col) location (tuple of int)

        """
        self.locations = {}
        for pop_dict in tqdm(self['populations'],
                             desc='Create GID/location mappings'):

            lay_name, pop_name = pop_dict['layer'], pop_dict['population']
            layer_gid = self['layers'][lay_name]['gid']

            # Create self.location[lay_name] dictionary if it doesn't exist.
            if lay_name not in self.locations:
                self.locations[lay_name] = {}

            self.locations[lay_name][pop_name] = gid_location_mapping(layer_gid,
                                                                      pop_name)
