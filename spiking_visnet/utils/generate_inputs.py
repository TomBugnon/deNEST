#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_inputs.py

from math import ceil
from os.path import join

import numpy as np

from .system import mkdir_ifnot


def vertical_cross(vsize=9, hsize=9, width=3):
    """ Returns vsize * hsize np.array containing a centered cross of width."""

    a = np.zeros((vsize, hsize))
    v_c, h_c = int(vsize / 2), int(hsize / 2)
    half_w = int((width - 1) / 2)

    a[:, h_c - half_w: h_c + half_w + 1] = 1
    a[v_c - half_w: v_c + half_w + 1, :] = 1

    return a


def vertical_tee(vsize=9, hsize=9, width=3):
    """ Returns vsize * hsize np.array containing a centered T of width."""

    a = np.zeros((vsize, hsize))
    h_c = int(hsize / 2)
    half_w = int((width - 1) / 2)

    a[:, h_c - half_w: h_c + half_w + 1] = 1
    a[0: width, :] = 1

    return a


FUN_MAP = {
    'vertical_cross': vertical_cross,
    'vertical_tee': vertical_tee
}


def create_movie(raw_input_dir, res, t, stim_type, path_type='default',
                 vsize=9, hsize=9, width=3, save=True):

    stim = FUN_MAP[stim_type](vsize, hsize, width)
    path = generate_path(res, t, path_type)
    mv = generate_movie(res, stim, path)

    # save
    savestr = generate_movie_str(stim_type, path_type, res, t, vsize, hsize,
                                 width)
    mkdir_ifnot(raw_input_dir)
    np.save(join(raw_input_dir, savestr), mv)
    return mv


def generate_movie_str(stim_type, path_type, res, t, vsize, hsize, width):
    return (stim_type + '_path=' + path_type + '_res=(' + str(res[0]) + ',' +
            str(res[1]) + ')_t=' + str(t) + '_vsize=' + str(vsize) + '_hsize=' +
            str(hsize) + '_width=' + str(width))


def generate_path(res, t, path_type='default'):
    """ Generate a path of the center of the stimulus. The default is a top to
    bottom x left to right path such that after t timesteps the stimulus is at
    the bottom right of the array.

    Args:
        - res (2-tuple): Dimension (x,y) of the image the path is generated for.
        - t (int): Number of time steps
        - path_type (str): 'default' -> left to right, top to bottom.

    Returns:
        - (t-list of tuples): [ (x_center(t), y_center(t), ...] for each
            timestep
    """

    N = np.prod(res)

    if path_type == 'default':
        a = np.zeros(N)
        # Distribute range(t) evenly in N*1-array
        a[evenly_spread_indices(N, t)] = np.arange(t) + 1
        # Reshape back to image size
        a = np.reshape(a, res)
        # Get coordinates
        return [find_in_np(a, i + 1) for i in range(t)]
    else:
        raise Exception('No code')


def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]


def evenly_spread_indices(N, num):
    """ Returns evenly spread indices for a list a size N"""
    return list(takespread(range(N), num))


def find_in_np(nparray, value):
    ''' Returns the coordinate tuple of the first encounter of value in array.
    '''
    return tuple(zip(*np.where(nparray == value)))[0]


def generate_movie(res, stim, path):

    tdim = len(path)
    a = np.zeros((res[0], res[1], tdim))
    for t in range(tdim):
        a[:, :, t] = warped_addition(np.zeros((res)),
                                     stim,
                                     path[t])

    return a


def warped_addition(abig, asmall, center):
    """ Add two arrays of different size, provided the 'abig' coordinate of the
    top left of <asmall>. <abig> is considered as wrapped.
    eg:
        - abig = np.zeros(4,4), asmall = ones(3,3), center = (0,0)
            returns [[1, 1, 1, 0],
                     [1, 1, 1, 0],
                     [1, 1, 1, 0],
                     [0, 0, 0, 0]]
        - abig = np.zeros(4,4), asmall = ones(3,3), center = (1,2)
            returns [[0, 0, 0, 0],
                     [1, 0, 1, 1],
                     [1, 0, 1, 1],
                     [1, 0, 1, 1]]

    Args:
        - abig, asmall (np.array)
        - center (2-tuple)
    Returns:
        - (np.array) of same dim as abig.
    """
    vdim_b, hdim_b = np.shape(abig)
    vdim_s, hdim_s = np.shape(asmall)
    # Increase dimensionality of asmall
    a = np.zeros((vdim_b, hdim_b))
    try:
        a[:vdim_s, :hdim_s] = asmall  # asmall is on 'top-left'
    except ValueError:
        print('asmall should have smaller dimensionality than abig.')

    # Roll the array so that it is properly aligned with abig
    a = np.roll(a, center[0], axis=0)
    a = np.roll(a, center[1], axis=1)

    return abig + a
