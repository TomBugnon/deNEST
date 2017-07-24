#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# downsample.py


import numpy as np
import scipy.misc


# TODO
def downsample(input_movie, preprocessing_params, network):
    """Downsample input_movie to fit with network's input resolution."""
    xdim, ydim = network.input_res()
    tdim = np.size(input_movie, axis=2)
    output_movie = np.zeros((xdim, ydim, tdim))
    for t in range(tdim):
        output_movie[:, :, t] = scipy.misc.imresize(input_movie[:, :, t],
                                                    (xdim, ydim))
    return output_movie


def get_string(_, network):
    """Return summary string of this preprocessing step."""
    (xdim, ydim) = network.input_res()
    return ('res_' + str(xdim) + 'x' + str(ydim) + '_')
