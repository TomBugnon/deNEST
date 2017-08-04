#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

from os import stat
from os.path import basename, exists, isfile, join, splitext
from shutil import rmtree

import nest
import numpy as np
import yaml
from tqdm import tqdm

from user_config import SAVE_DIR

from .utils.format_recorders import format_mm_data, format_sd_data
from .utils.sparsify import save_as_sparse

FULL_PARAMS_TREE_STR = 'params.yaml'
NETWORK_STR = 'network.yaml'
SIM_METADATA_STR = 'metadata.yaml'
STRING_SEPARATOR = '-__-'


def save_as_yaml(path, tree):
    """Save <tree> as yaml file at <path>."""
    with open(path, 'w') as f:
        yaml.dump(tree, f, default_flow_style=False)


def load_yaml(*args):
    """Load yaml file from joined (os.path.join) arguments.

    Return empty list if the file doesn't exist.
    """
    file = join(*args)
    if exists(file):
        with open(join(*args), 'rt') as f:
            return yaml.load(f)
    else:
        return []


def save_all(network, full_params_tree):
    """Save all network and simulation information.

    - the full parameter tree use to define the network and the simulation,
    - The full network object and the full architecture in NEST, including
        connections.
    - the formatted activity recorded by the recorders that should be saved
        (defined in network.populations).
    - The simulation information (containing eg: 'ms per timestep' and session
        information)

    Args:
        - <network> (Network): The nest-initialized network.
        - <full_params_tree> (dict): Original full parameter tree defining the
            network and the simulation.

    """

    # Get target directories for formatting.
    sim_savedir = get_simulation_savedir(network)
    # Create if not already done
    mkdir_ifnot(sim_savedir)

    print(f'Save everything in {sim_savedir}')

    # Save full params
    print('Save parameters.')
    save_as_yaml(join(sim_savedir, FULL_PARAMS_TREE_STR), full_params_tree)

    # TODO: Save network
    print('Save full network.')
    network.save(join(sim_savedir, NETWORK_STR))

    # Save recorders
    print('Save recorders.')
    save_formatted_recorders(network, sim_savedir)
    # Delete temporary recorder dir
    if full_params_tree['children']['simulation']['delete_tmp_dir']:
        rmtree(get_NEST_tmp_savedir(network, user_savedir=user_savedir))

    # TODO: Save simulation data
    print('Save simulation metadata.')
    save_simulation()


def save_simulation():
    """Save simulation metadata (time, session at each timestep, etc)."""
    pass


def generate_save_subdir_str(full_params_tree, param_file_path):
    """Create and return relative path to the simulation saving directory.

    Returns:
        (str): The full path to the simulation saving directory  will be
            SAVE_DIR/subdir_str

    """
    # For now, use only the filename without extension of the parameter file.
    param_file = splitext(basename(param_file_path))[0]
    subdir_str = param_file
    return subdir_str


def get_simulation_savedir(network):
    """Return absolute path to directory in which we save formatted sim data."""
    return join(SAVE_DIR, network.save_subdir_str)


def get_NEST_tmp_savedir(network):
    """Return absolute path to directory in which NEST saves recorder data."""
    return join(SAVE_DIR, network.save_subdir_str, 'tmp')


def save_formatted_recorders(network, sim_savedir):
    """Format all networks' recorder data.

    The format of the filenames in the saving directory for each population and
    each recorded variable (eg: 'V_m', 'spike', 'g_exc', ...) is:
        (<layer_name> + STRING_SEPARATOR + <population_name> + STRING_SEPARATOR
        + <variable_name>)

    Args:
        - network (Network object)
        - recorder_tmp_savedir (str): path to directory in which NEST saved all
            the non-formatted recorder data
        - sim_savedir (str): path to directory in which we will save all the
            formatted recorder data

    """
    population_list = network['populations']
    gid_location_mappings = network.locations

    # For (ntimesteps * nrow * ncol)-nparray initialization
    n_timesteps = int(nest.GetKernelStatus('time')
                      / nest.GetKernelStatus('resolution'))

    for pop_dict in tqdm(population_list,
                         desc='--> Format recorder data'):

        layer = pop_dict['layer']
        pop = pop_dict['population']
        mm = pop_dict['mm']
        sd = pop_dict['sd']
        location_by_gid = gid_location_mappings[layer][pop]['location']

        # For layer size for (total_time * nrow * ncol)-nparray initialization
        layer_params = network['layers'][layer]['nest_params']
        (nrow, ncol) = layer_params['rows'], layer_params['columns']

        # Population string for saving filename
        pop_string = layer + STRING_SEPARATOR + pop + STRING_SEPARATOR

        if mm['gid']:

            recorded_variables = nest.GetStatus(mm['gid'], 'record_from')[0]

            for variable in [str(var) for var in recorded_variables]:

                time, gid, activity = gather_raw_data(mm['gid'],
                                                      variable,
                                                      recorder_type='multimeter'
                                                      )
                activity_array = format_mm_data(gid,
                                                time,
                                                activity,
                                                location_by_gid,
                                                dim=(n_timesteps, nrow, ncol))
                filename = pop_string + variable
                save_as_sparse(join(sim_savedir, filename),
                               activity_array)

        if sd['gid']:
            time, gid = gather_raw_data(sd['gid'],
                                        recorder_type='spike_detector')

            activity_array = format_sd_data(gid,
                                            time,
                                            location_by_gid,
                                            dim=(n_timesteps, nrow, ncol))
            filename = pop_string + 'spikes'

            save_as_sparse(join(sim_savedir, filename),
                           activity_array)


def gather_raw_data(rec_gid, variable='V_m', recorder_type=None):
    """Return non-formatted activity of a given variable saved by the recorder.

    Args:
        - <rec_gid> (tuple): Recorder's NEST GID. Singleton tuple of int.
        - <variable> (str): Variable recorded that we return. Used only for
            multimeters.
        - <recorder_type> (str): 'multimeter' or 'spike_detector'

    Returns:
        (tuple): Tuple of 1d np.arrays of the form
            - (<time>, <sender_gid>, <activity>) for a multimeter, where
                activity is that of the variable < variable >.
            - (<time>, <sender_gid>) for a spike detector.

    """
    record_to = nest.GetStatus(rec_gid, 'record_to')[0]

    if 'memory' in record_to:

        data = nest.GetStatus(rec_gid, 'events')[0]
        time = data['times']
        sender_gid = data['senders']

        if recorder_type == 'multimeter':
            activity = data[variable]
            return (time, sender_gid, activity)
        elif recorder_type == 'spike_detector':
            return (time, sender_gid)

    elif 'file' in record_to:

        recorder_files = nest.GetStatus(rec_gid, 'filenames')[0]

        data = load_and_combine(recorder_files)
        time = data[:, 1]
        sender_gid = data[:, 0]

        if recorder_type == 'multimeter':
            # Get proper column
            all_variables = nest.GetStatus(rec_gid, 'record_from')[0]
            variable_col = 2 + all_variables.index(variable)
            activity = data[:, variable_col]
            return (time, sender_gid, activity)
        elif recorder_type == 'spike_detector':
            return (time, sender_gid)


def load_and_combine(recorder_files_list):
    """Load the recorder data from files.

    Args:
        - <recorder_files_list> (list): List of absolute paths to the files in
            which NEST saved a single recorder's activity.

    Returns:
        (np.array): Array of which columns are the files' columns and rows are
            the events recorded in the union of all files. If all the files are
            empty or there is no filename, returns an array with 0 rows.
            Array np-loaded in each text file is enforced to have two
            dimensions. If no data is found at all, returns an array with zero
            rows.

    """
    file_data_list = [np.loadtxt(filepath, dtype=float, ndmin=2)
                      for filepath in recorder_files_list
                      if isfile(filepath) and not (stat(filepath).st_size == 0)]

    if file_data_list:
        return np.concatenate(file_data_list, axis=0)
    else:
        return np.empty((0, 10))
