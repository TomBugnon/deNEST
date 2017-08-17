#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# downsample.py


"""Resize movie frames to network's resolution."""

import numpy as np
from scipy.ndimage import interpolation


# TODO
def resize(input_movie, preprocessing_params, network):
    """Resize input_movie to fit with network's input resolution."""
    xdim_t, ydim_t = network.input_res()
    xdim_s, ydim_s = np.size(input_movie, 1), np.size(input_movie, 2)
    tdim = np.size(input_movie, axis=0)
    output_movie = np.zeros((tdim, xdim_t, ydim_t))
    for t in range(tdim):
        output_movie[t, :, :] = interpolation.zoom(input=input_movie[t, :, :],
                                                   zoom=(float(xdim_t/xdim_s),
                                                         float(ydim_t/ydim_s)),
                                                   order=0)
    return output_movie


def get_string(_, network):
    """Return summary string of this preprocessing step."""
    (xdim, ydim) = network.input_res()
    return 'res_' + str(xdim) + 'x' + str(ydim) + '_'
