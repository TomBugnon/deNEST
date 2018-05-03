#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_inputs.py
"""Generate and save movies (3D nparrays) by moving a stimulus across frames."""

import itertools
import math
from math import ceil
from os import makedirs
from os.path import join

import numpy as np

from ..save import save_array

# pylint: disable=missing-docstring,too-many-arguments,invalid-name

def vertical_cross(nrows=9, ncols=9, width=3):
    """Return nrows * ncols np.array containing a centered cross of width."""
    array = np.zeros((nrows, ncols))
    v_c, h_c = int(nrows / 2), int(ncols / 2)
    half_w = int((width - 1) / 2)

    array[:, h_c - half_w:h_c + half_w + 1] = 1
    array[v_c - half_w:v_c + half_w + 1, :] = 1

    return array


def vertical_tee(nrows=9, ncols=9, width=3):
    """Return nrows * ncols np.array containing a centered T of width."""
    array = np.zeros((nrows, ncols))
    h_c = int(ncols / 2)
    half_w = int((width - 1) / 2)

    array[:, h_c - half_w:h_c + half_w + 1] = 1
    array[0:width, :] = 1

    return array


def vertical_L(nrows=9, ncols=9, width=3):
    """Return nrows * ncols np.array containing a centered + of width."""
    array = np.zeros((nrows, ncols))
    h_c = int(ncols / 2)
    half_w = int((width - 1) / 2)

    array[:, :width] = 1
    array[nrows - width:nrows, :] = 1

    return array


FUN_MAP = {
    'vertical_cross': vertical_cross,
    'vertical_tee': vertical_tee
}


def sinusoidal_grating(nrows=9, ncols=9, mean=0.5, amplitude=0.5, phase=0,
                       spatial_period=None, angle=0):
    if spatial_period is None:
        spatial_period = min(nrows, ncols) / 4
    array = np.zeros((nrows, ncols))
    normal_vector = math.cos(angle), math.sin(angle)
    for row, col in itertools.product(range(nrows), range(ncols)):
        # translate np coordinate into trigonometric
        distance = (
            normal_vector[0] * (1 * col) + normal_vector[1] * (-1 * row))
        sin_variation = math.sin(2 * math.pi *
                                 (distance - phase) / spatial_period)
        array[row, col] = max(0, mean + amplitude * sin_variation)
    return array


def create_movie(raw_input_path, res, t, stim_type, path_type='default', nrows=9,
                 ncols=9, width=3, save=False):
    """Create, possibly save, and return a movie (3D np-array).

    Args:
        raw_input_path (str): path to the `raw_input` subdirectory of the
            USER's input directory
        res (tuple): Dimension of each frame.
        t (int): Number of frames.
        stim_type (str): Type of the moving stimulus. Defines which function
            is called (see FUN_MAP)
        path_type (str): Defines the type of path the stimulus 'takes' across
            frames of the movie
        nrows (int): vertical size (first dimension) of the stimulus
        ncols (int): horizontal size (second dimension) of the stimulus
        width (int): width of the stimulus
        save (bool): If true, saves the created movie in INPUT_PATH/raw_inputs

    Returns:
        np-array: (nframes*nrows*ncols)-numpy array

    """
    stim = FUN_MAP[stim_type](nrows, ncols, width)
    path = generate_path(res, t, path_type)
    movie = generate_movie(res, stim, path)

    if save:
        savestr = generate_movie_str(stim_type, path_type, res, t, nrows, ncols,
                                     width)
        makedirs(raw_input_path, exist_ok=True)
        save_array(join(raw_input_path, savestr), movie)

    return movie


def generate_movie_str(stim_type, path_type, res, t, nrows, ncols, width):
    """Generate the filename under which a movie is saved."""
    return (stim_type + '_path=' + path_type + '_res=(' + str(res[0]) + ',' +
            str(res[1]) + ')_t=' + str(t) + '_nrows=' + str(nrows) + '_ncols=' +
            str(ncols) + '_width=' + str(width))


def generate_path(res, t=None, path_type='default'):
    """Generate a path across frames of the top-left corner of the stimulus.

    The default is a (left to right) x (top to bottom) path such that after t
    timesteps the stimulus is at the bottom right of the array.

    Args:
        res (tuple): Dimension of each frame.
        t (int): Number of time steps/frames
        path_type (str): 'default' -> left to right, top to bottom.

    Returns:
        list: List of tuples defining the position of the top-left corner of
            the stimulus at each time-step. [ (x_topleft(t), y_topleft(t), ...]

    """
    nrows, ncols = res
    Nelems = np.prod(res)

    if path_type == 'default':
        a = np.zeros(Nelems)
        # Distribute range(t) evenly in N*1-array
        a[evenly_spread_indices(N, t)] = np.arange(t) + 1
        # Reshape back to image size
        a = np.reshape(a, res)
        # Get coordinates
        return [find_in_np(a, i + 1) for i in range(t)]
    elif path_type == 'top_left_to_top_right':
        return [(0, col) for col in range(ncols)]
    elif path_type == 'top_left_to_bottom_left':
        return [(row, 0) for row in range(nrows)]
    elif path_type == 'Z':
        # t = 9
        # Three top positions, three middle, three bottom
        return [(int(i * res[0] / 3.), int(j * res[0] / 3.))
                for (i, j) in itertools.product(range(3), range(3))]

    else:
        raise Exception('No code')


def takespread(sequence, num):
    """Return `num` elements of the list `sequence` that are evenly spread."""
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]


def evenly_spread_indices(N, num):
    """Return evenly spread indices for a list a size N.

    Used to define default path.
    """
    return list(takespread(range(N), num))


def find_in_np(nparray, value):
    """Return the coordinate tuple of the first find of value in array."""
    return tuple(zip(*np.where(nparray == value)))[0]


def generate_movie(res, stim, path):
    """Create a movie from a stimulus and a path.

    Each frame (time t) is defined by inserting the array `stim` at the position
    defined by path(t) in an array of zeros of size `res`
    """
    tdim = len(path)
    a = np.zeros((tdim, res[0], res[1]))
    for t in range(tdim):
        a[t, :, :] = wrapped_addition(np.zeros((res)), stim, path[t])
    return a


def wrapped_addition(abig, asmall, pos):
    """Add a smaller array to a larger one with wrapping.

    Because the two arrays have different size, the coordinates of the top-left
    of the small array are given by `pos`.

    Args:
        abig (np.ndarray): 'big' array
        asmall (np.ndarray): 'small' array
        pos (tuple): coordinate of the 'big array' at which we place the 'small'
            array to perform the addition

    Examples:
        >>> abig = np.zeros(4,4)
        >>> asmall = ones(3,3)
        >>> center = (0,0)
        >>> wrapped_addition(abig, asmall, pos)
        [[1, 1, 1, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]]
        >>> abig = np.zeros(4,4)
        >>> asmall = ones(3,3)
        >>> pos = (1,2)
        >>> wrapped_addition(abig, asmall, pos)
        [[0, 0, 0, 0],
         [1, 0, 1, 1],
         [1, 0, 1, 1],
         [1, 0, 1, 1]]

    Returns:
        np.array: of same dim as abig.

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
    a = np.roll(a, pos[0], axis=0)
    a = np.roll(a, pos[1], axis=1)

    return abig + a
