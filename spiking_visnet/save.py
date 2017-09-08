#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save.py


"""Save and load movies, networks, activity and simulation parameters."""

import itertools
import os
from os import makedirs, stat
from os.path import basename, exists, isfile, join, splitext
from shutil import rmtree

import matplotlib.pyplot as plt
import nest
import numpy as np
import pylab
import yaml
from nest import raster_plot
from tqdm import tqdm

from user_config import OUTPUT_DIR

from .nestify.connections import get_filtered_synapses
from .utils.format_recorders import format_mm_data, format_sd_data
from .utils.sparsify import load_as_numpy, save_array

FULL_PARAMS_TREE_STR = 'params.yaml'
NETWORK_STR = 'network.yaml'
SIM_METADATA_STR = 'metadata.yaml'
STRING_SEPARATOR = '_'


def load_session_times(output_dir):
    """Load session time from output dir."""
    return load_yaml(output_dir, 'session_times')


def load_session_stim(output_dir, session_name):
    """Load full stimulus of a session."""
    full_stim_filename = 'session_' + session_name + '_full_stim.npy'
    return load_as_numpy(join(output_dir, full_stim_filename))


def load_activity(output_dir, layer, population, variable='spikes',
                  session=None, all_units=False):
    """Load activity of a given variable for a population."""
    if all_units:
        filename_prefix = recorder_filename(layer, population,
                                            variable=variable, unit_index=None)
    else:
        filename_prefix = recorder_filename(layer, population,
                                            variable=variable, unit_index=0)
    all_filenames = [f for f in os.listdir(output_dir)
                     if f.startswith(filename_prefix)
                     and isfile(join(output_dir, f))]

    # Concatenate along first dimension (row)
    all_sessions_activity = np.concatenate(
        [load_as_numpy(join(output_dir, filename))
         for filename in all_filenames],
        axis=1
        )
    if session is None:
        return  all_sessions_activity
    session_times = load_session_times(output_dir)
    return all_sessions_activity[session_times[session]]


def load_labels(output_dir, session_name):
    """Load labels of a session."""
    labels_filename = 'session_' + session_name + '_labels.npy'
    return np.load(join(output_dir, labels_filename))


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


def save_all(network, simulation, full_params_tree):
    """Save all network and simulation information.

    - the full parameter tree use to define the network and the simulation,
    - The full network object and the full architecture in NEST, including
        connections.
    - the formatted activity recorded by the recorders that should be saved
        (defined in network.populations).
    - The simulation information (containing eg: 'ms per timestep' and session
        information)
    - The session stimuli ()

    Args:
        network (Network): The nest-initialized network.
        simulation (Simulation): Simulation object. Contains Sessions
        full_params_tree (Params): Parameter object
    """
    # Get relevant part of the full param tree.
    sim_params = full_params_tree['children']['simulation']

    # Get target directories for formatting.
    sim_savedir = get_simulation_savedir(network, sim_params)
    # Create if not already done
    makedirs(sim_savedir, exist_ok=True)

    print(f'Save everything in {sim_savedir}')

    # Save nest raster plots
    print('Save nest raster plots.')
    if sim_params.get('save_nest_raster', False):
        save_nest_raster(network, sim_savedir)

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
    if sim_params.get('delete_tmp_dir', True):
        rmtree(get_nest_tmp_savedir(network, sim_params))

    print('Save synapses.')
    # Save synapses' parameters.
    save_synapses(network, sim_savedir)

    # TODO: Save simulation data
    print('Save simulation metadata.')
    save_simulation()

    # Save sessions stimuli
    print('Save sessions stimuli')
    simulation.save_sessions(sim_savedir)


def save_nest_raster(network, output_dir):
    """Use NEST's raster function to save activity pngs.

    Only do so for recorders saved on memory for which there were spikes.
    """
    for pop in tqdm(network['populations'],
                    desc='--> Save nest raster plots'):
        rec_gid = pop['sd']['gid']
        if (rec_gid
            and 'memory' in pop['sd']['rec_params']['record_to']):
            # Check there is at least one event to avoid NESTError
            if len(nest.GetStatus(rec_gid)[0]['events']['senders']):
                raster = raster_plot.from_device(rec_gid, hist=True)
            else:
                print("Didn't generate raster plot since there were no spikes.")
                continue
            pylab.title(pop['layer']+'_'+pop['population'])
            f = raster[0].figure
            f.set_size_inches(15, 9)
            filename = ('spikes_raster_' + pop['layer'] + '_'
                        + pop['population'] + '.png')
            f.savefig(join(output_dir, filename), dpi=100)
            plt.close()


def save_synapses(network, sim_savedir):
    """Save synapses' values.

    For each connection and each queried keys in single synapses' dictionary,
    saves a np-array of size::
                (``nrows_target``, ``ncols_target``,
                 ``nrows_source``, ``ncols_source``)
    Of which the value at indices (row_t, col_t, row_s, col_s) is the value of
    a specific entry (eg: 'weight') for the synapse of a given
    population-to-population connection between an arbitrary pre_synaptic unit
    at location (row_s, col_s) and an arbitrary post-synaptic unit at location
    (row_t, col_t).

    TODO:
        * For plastic synapses, force a single spike in all neurons to account
            for synapses' deferred update.

    """
    locations = network.locations

    for connection in tqdm(network['connections'],
                           desc='--> Format synapses data.'):

        # Connection string for saving filename
        base_connection_string = make_base_connection_string(connection)

        # Access source and target unit locations
        source_locs = locations[connection['source_layer']
                                ][connection['source_population']]
        target_locs = locations[connection['target_layer']
                                ][connection['target_population']]

        keys = connection['params'].get('save', [])
        if keys:
            # (nrows_target, ncols_target, nrows_source, ncols_source, nkeys)
            # np-array.
            all_synapses_all_values = get_synapses_values(network,
                                                          connection,
                                                          source_locs,
                                                          target_locs,
                                                          saving_keys=keys)

            # Save the different types of values in separate arrays
            for i, key in enumerate(keys):
                save_array(join(sim_savedir,
                                    base_connection_string + '_' + key),
                               all_synapses_all_values[:, :, :, :, i])


def get_synapses_values(network, connection, source_locs, target_locs,
                        saving_keys=None):
    """Return an array containing the weight of a connection by location.

    Args:
        network (Network): Initialized network object,
        connection (dict): Dict describing a specific population-to-population
            connection,
        source_locs (dict): Bi-directional location/GID mappings of units in the
            topological connection's source population,
        target_locs (dict): Bi-directional location/GID mappings of units in the
            topological connection's target population,
        saving_keys (list of str): List of keys of which we return the values in
            each single synapses' dictionary.

    Returns:
        np-array: Array of size::
                (``nrows_target``, ``ncols_target``,
                 ``nrows_source``, ``ncols_source``,
                 ``nkeys``)
            where:
            - (``nrows_..``, ``ncols_target..``) is the resolution of the
            post-synaptic or pre_synaptic layers for the connection considered,
            - ``nkeys`` is the number of keys queried in each single synapse's
            dictionary.
            The value at indices (row_t, col_t, row_s, col_s, i) is the value
            of the i_th key of ``saving_keys`` in the synapse connecting a
            target unit at location (row_t, col_t) in the target layer and a
            unit at location (row_s, col_s) in the source layer.

            NB: - If multiple elements of the same population coexist at
            the same location in either the target or the source layer, the
            weights for only one of them (chosen arbitrarily) is returned.

    TODO:
        * Optimize for speed!

    """
    # Access useful values in connection dict.
    source_lay = connection['source_layer']
    source_pop = connection['source_population']
    target_lay = connection['target_layer']
    source_lay_gid = network['layers'][source_lay]['gid']

    # Get layers' resolution
    nrows_target = network['layers'][target_lay]['nest_params']['rows']
    ncols_target = network['layers'][target_lay]['nest_params']['columns']
    nrows_source = network['layers'][source_lay]['nest_params']['rows']
    ncols_source = network['layers'][source_lay]['nest_params']['columns']

    # Initialize return array with -999 (absense of synapse).
    all_synapses_all_values = -999 * np.ones((nrows_target, ncols_target,
                                            nrows_source, ncols_source,
                                            len(saving_keys)))

    # Iterate on target layer to fill return array.
    for row_target, col_target in itertools.product(range(nrows_target),
                                                    range(ncols_target)):

        # Reference unit at a given location (Arbitrary choice after filtering
        # for population
        target_gid = target_locs['gid'][row_target, col_target, 0]

        # Get incoming synapses from for that unit
        filtered_synapses = get_filtered_synapses(
            target_gid, ref='target', secondary_layer_gid=source_lay_gid[0],
            secondary_population=source_pop)

        # Gather values for all the queried keys for each synapse.
        for synapse in filtered_synapses:
            # Get source location
            source_gid = synapse['source']
            row_source, col_source, _ = source_locs['location'][source_gid]
            # Get values of interest for that synapse
            synapse_values = [synapse[key] for key in saving_keys]

            # Fill array with synapses' values in each of the keys.
            all_synapses_all_values[row_target, col_target,
                                    row_source, col_source, :] = synapse_values

    return all_synapses_all_values


def make_base_connection_string(connection):
    """Generate string describing a population-to-population connection."""
    return ('synapses_from_' + connection['source_layer'] + STRING_SEPARATOR
            + connection['source_population'] + '_to_'
            + connection['target_layer'] + STRING_SEPARATOR
            + connection['target_population'])


def save_simulation():
    """Save simulation metadata (time, session at each timestep, etc)."""
    pass


def generate_save_subdir_str(network_params, sim_params):
    """Create and return relative path to the simulation saving directory.

    Returns:
        str: If not specified manually by USER, the full path to the
            simulation saving directory  will be OUTPUT_DIR / subdir_str

    """
    # For now, use only the filename without extension of the full parameter
    # file.
    param_file = splitext(basename(sim_params['param_file_path']))[0]
    subdir_str = param_file
    return subdir_str


def get_simulation_savedir(network, sim_params):
    """Return absolute path to directory in which we save formatted sim data.

    Either defined by user('user_savedir' key in sim_params) or an
    automatically generated subdirectory of OUTPUT_DIR.

    """
    if not sim_params.get('user_savedir', None):
        return join(OUTPUT_DIR, network.save_subdir_str)
    return sim_params['user_savedir']


def get_nest_tmp_savedir(network, sim_params):
    """Return absolute path to directory in which NEST saves recorder data.

    Nest saves in the 'tmp' subdirectory of the simulation saving directory.
    """
    return join(get_simulation_savedir(network, sim_params),
                'tmp')


def save_formatted_recorders(network, sim_savedir):
    """Format all networks' recorder data.

    The format of the filenames in the saving directory for each population and
    each recorded variable(eg: 'V_m', 'spike', 'g_exc', ...) is:
        ( < layer_name > + STRING_SEPARATOR + <population_name > + STRING_SEPARATOR
        + <variable_name > )

    NB: For each population, we only save the activity of one unit at each
    location. Remember there can be multiple units at each location for each
    population.

    Args:
        network(Network object)
        sim_savedir(str): path to directory in which we will save all the
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

        # Get layer size for (total_time * nrow * ncol)-nparray initialization
        layer_params = network['layers'][layer]['nest_params']
        (nrow, ncol) = layer_params['rows'], layer_params['columns']


        # Iterate on unit index (there can be multiple units per location)
        nunits_per_location = np.size(gid_location_mappings[layer][pop]['gid'],
                                      axis=2)
        for unit_index in range(nunits_per_location):

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
                                                    dim=(n_timesteps, nrow, ncol),
                                                    unit_index=unit_index)
                    filename = recorder_filename(layer, pop,
                                                 unit_index=unit_index,
                                                 variable=variable)
                    save_array(join(sim_savedir, filename),
                                   activity_array)

            if sd['gid']:
                time, gid = gather_raw_data(sd['gid'],
                                            recorder_type='spike_detector')
                activity_array = format_sd_data(gid,
                                                time,
                                                location_by_gid,
                                                dim=(n_timesteps, nrow, ncol),
                                                unit_index=unit_index)
                filename = recorder_filename(layer, pop,
                                             variable='spikes',
                                             unit_index=unit_index)
                save_array(join(sim_savedir, filename),
                               activity_array)


def recorder_filename(layer, pop, unit_index=None, variable='spikes'):
    """Return filename for a population x unit_index."""
    base_filename = (layer + STRING_SEPARATOR + pop + STRING_SEPARATOR
                     + variable)
    suffix = ''
    if unit_index is not None:
        suffix = (STRING_SEPARATOR + 'units' + STRING_SEPARATOR
                  + str(unit_index))
    return base_filename + suffix


def gather_raw_data(rec_gid, variable='V_m', recorder_type=None):
    """Return non - formatted activity of a given variable saved by the recorder.

    Args:
        rec_gid(tuple): Recorder's NEST GID. Singleton tuple of int.
        variable(str): Variable recorded that we return. Used only for
            multimeters.
        recorder_type(str): 'multimeter' or 'spike_detector'

    Returns:
        tuple: Tuple of 1d np.arrays of the form
            - ( < time > , < sender_gid > , < activity > ) for a multimeter, where
                activity is that of the variable < variable > .
            - (< time > , < sender_gid > ) for a spike detector.

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
        # from IPython.core.debugger import Tracer; Tracer()()
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
        recorder_files_list(list): List of absolute paths to the files in
            which NEST saved a single recorder's activity.

    Returns:
        (np.array): Array of which columns are the files' columns and rows are
            the events recorded in the union of all files. If all the files are
            empty or there is no filename, returns an array with 0 rows.
            Array np - loaded in each text file is enforced to have two
            dimensions. If no data is found at all, returns an array with zero
            rows.

    """
    file_data_list = [np.loadtxt(filepath, dtype=float, ndmin=2)
                      for filepath in recorder_files_list
                      if isfile(filepath) and not stat(filepath).st_size == 0]

    if file_data_list:
        return np.concatenate(file_data_list, axis=0)
    return np.empty((0, 10))
