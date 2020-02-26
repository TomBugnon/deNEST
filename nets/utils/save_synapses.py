#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# save_synapses.py


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
        source_locs = locations[connection['source_layer']][connection[
            'source_population']]
        target_locs = locations[connection['target_layer']][connection[
            'target_population']]

        keys = connection['params'].get('save', [])
        if keys:
            # (nrows_target, ncols_target, nrows_source, ncols_source, nkeys)
            # np-array.
            all_synapses_all_values = get_synapses_values(
                network, connection, source_locs, target_locs, saving_keys=keys)

            # Save the different types of values in separate arrays
            for i, key in enumerate(keys):
                save_array(
                    join(sim_savedir, base_connection_string + '_' + key),
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
    for row_target, col_target in itertools.product(
            range(nrows_target), range(ncols_target)):

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
            all_synapses_all_values[row_target, col_target, row_source,
                                    col_source, :] = synapse_values

    return all_synapses_all_values
