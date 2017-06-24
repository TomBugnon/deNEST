import nest


def init_spike_times(movie, network):
    """ Communicates to the kernel the spike times for all input neurons during
    the forecoming period.
    """

    spike_times = generate_spike_times(movie, network)
    for (gid, times) in spike_times:
        nest.SetStatus(gid, {'spike_times': spike_times})

    return


def generate_spike_times(session_params, network):
    """ Return for each input neuron some spike times drawn from a poisson
    distribution of instantaneous rate defined by the value corresponding to
    that neuron/timestep in the session's full stimulus.


    Args:
        - <session_params>: entry of the formatted 'session' tree.
        - <network> (dict): Output of the get_Network() functions. Useful keys
            in this function are 'areas' and 'layers'. Nest GIDs must be
            available (run init_Network() first)

    Returns:
        - (list): List of (<gid>, <spike_times>) tuples.
            - <gid> is a singleton tuple
            - <spike_times> is a list of spike times (in ms, from the start of
                the simulation run) for the neuron defined by <gid>.
    """

    return



"""
        <movie> (dict): dict containing the preprocessed movie and its
            meta-parameters.
                {'mv_params': {'filter_suffixes': <suff_list>,
                                ...}
                 'mv_array' : <mv_array>}
             where <mv_array> is a np.array of dimension (N * N * nlayers *
             ntimesteps). The values correspond the instantaneous poisson rates
             (determined during preprocessing). The dimensions correspond to:
             - N*N is the size of a single input layer,
             - nlayers is the number of input layers corresponding to the
                different filterings of a natural image. If there is only one
                input layer (retina), <suff_list> = ''.
                If there is more than one such layers, the frames are associated
                to the layers using the order in <suff_list>.
                    eg: N * N * 0 * t -> 'input_layer'+<suff_list>[0] etc
            - ntimesteps is the total duration of the period. This value can
                differ from the number of frames of the original input movie as
                it might have been trimmed, cycled, and each frame might have
                been repeated during preprocessing.
                """
