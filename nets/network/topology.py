#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/topology.py
"""Draw topological conns."""

import math
import random

import scipy.spatial


def draw_pool_gids(conn, driver_gid, N=1):  # pylint:disable=invalid-name
    """Draw n pool gids for a driver gid, taking in account topological params.

    Algorithm:
    - Get a list of candidate gids of the pool layer (checking population,
      mask and autapse)
    - While the number of drawn pool gids is smaller than the requested number:
        1. Check that the list of candidate gids is not empty
        1. Shuffle the list of candidate gids
        2. For each candidate gid in turn from the shuffled list, draw a success
           according to the point kernel probability.
           - if success, add gid to the return list and increase the count.
           - if not allow_multapse, pop the drawn gid from the list of
             candidate gids
        3. Return when the count is reached
    """
    # Get all pool gids and distance from driver (accounting for edge_wrap)
    all_gids = [(gid,
                 wrapped_distance(conn,
                                  conn.driver_layer.position(driver_gid)[0],
                                  conn.pool_layer.position(gid)[0]))
                for gid in conn.pool_gids()]
    # Filter out depending on mask
    candidate_gids = [(gid, distance)
                      for gid, distance in all_gids
                      if within_mask(conn, distance)]
    # Filter out autapse
    if not conn.nest_params.get('allow_autapses', True):
        candidate_gids = [tup for tup in candidate_gids if tup[0] != driver_gid]

    n_drawn = 0
    pool_gids = []
    while n_drawn < N:
        if not candidate_gids:
            raise Exception('Can not draw the requested number of gids... '
                            'Check mask/allow_multapses/allow_autapses params?')
        # Shuffle and draw according to point probability
        for gid, distance in random.sample(candidate_gids, len(candidate_gids)):
            if random.uniform(0, 1) <= kernel_probability(conn, distance):
                pool_gids.append(gid)
                n_drawn += 1
                if not conn.nest_params.get('allow_multapses'):
                    candidate_gids = [
                        tup for tup in candidate_gids if tup[0] != gid
                    ]
            if n_drawn == N:
                break
    return pool_gids


def within_mask(conn, distance):
    """Check whether a pool unit is within the mask."""
    # TODO: check allow_oversized_mask
    mask = conn.nest_params.get('mask', None)
    if not mask:
        return True
    elif 'circular' in mask:
        return distance < mask['circular']['radius']
    raise NotImplementedError


def wrapped_distance(conn, driver_position, pool_position):
    """Return the distance between driver and pool unit, considering wrap."""
    if not conn.pool_layer.params['edge_wrap']:
        return scipy.spatial.distance.euclidean(driver_position, pool_position)
    x_d, y_d = driver_position
    x_p, y_p = pool_position
    # TODO: IMPORTANT: Deal with case where the driver position is outside
    # of the pool layer
    # TODO: Make sure layers are centered
    pool_extent = conn.pool_layer.extent
    assert abs(x_d) <= pool_extent[0] / 2
    assert abs(y_d) <= pool_extent[1] / 2
    return math.sqrt(
        min(abs(x_d - x_p), pool_extent[0] - abs(x_d - x_p))**2 +
        min(abs(y_d - y_p), pool_extent[1] - abs(y_d - y_p))**2)


def kernel_probability(conn, distance):
    """Return the point probability of connection for a pool-target distance."""
    kernel = conn.nest_params.get('kernel', None)
    if kernel is None:
        return 1.
    elif isinstance(kernel, float):
        return kernel
    elif not kernel:
        # empty dict
        return 1.
    elif 'gaussian' in kernel:
        if 'anchor' in kernel['gaussian']:
            raise NotImplementedError
        cutoff = kernel['gaussian'].get('cutoff', 0.)
        sigma = kernel['gaussian']['sigma']
        p_center = kernel['gaussian']['p_center']
        return cutoff + p_center * math.exp(-(distance / sigma)**2 / 2)
    raise NotImplementedError
