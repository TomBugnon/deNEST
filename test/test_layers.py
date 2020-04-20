#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_layers.py


"""Test ``Layer`` class."""

import nest
import pytest

from nets.network.layers import InputLayer, Layer
from nets.utils.validation import ParameterError

BASE_LAYERS = [
    (
        Layer,
        {'populations': {'iaf_psc_alpha': 1}},
        {'rows': 1, 'columns': 1},
    ),
    (
        Layer,
        {'populations': {'iaf_psc_alpha': 2, 'iaf_cond_alpha': 2}},
        {'rows': 2, 'columns': 2, 'edge_wrap': True},
    ),
]

INPUT_LAYERS = [
    (
        InputLayer,
        {'populations': {'poisson_generator': 1}},
        {'rows': 2, 'columns': 2},
    ),
]


BAD_LAYERS = [
    (
        Layer,
        {},
        {'rows': 1, 'columns': 1},
    ),
    (
        Layer,
        {'populations': {'iaf_psc_alpha': 1}},
        {'columns': 1},
    ),
    (
        Layer,
        {'populations': {'iaf_psc_alpha': 1}},
        {'rows': 1},
    ),
    (
        Layer,
        {'populations': {'iaf_psc_alpha': 1}},
        {'rows': 1, 'columns': 1, 'elements': 'iaf_psc_alpha'},
    ),
    (
        InputLayer,
        {'populations': {'poisson_generator': 2}},
        {'rows': 1, 'columns': 1},
    ),
]


def init_layer(constructor, params, nest_params):
    layer = constructor('', params, nest_params)
    yield layer


def test_layer(layer):
    layer.create()
    # Correct gids
    assert set(nest.GetLeaves(layer.gid)[0]) == set(layer.gids())
    for population, pop_number in layer.populations.items():
        # Correct model
        assert all(
            [
                nest.GetStatus((gid,), 'model')[0] == population
                for gid in layer.gids(population=population)
            ]
        )
        # Correct number of units at each location
        assert all(
            [
                len(
                    layer.gids(
                        population=population,
                        location=location
                    )
                ) == pop_number
                for location in layer
            ]
        )


@pytest.fixture(params=BAD_LAYERS)
def bad_layer(request):
    yield request.param


def test_bad_layer(bad_layer):
    constructor, params, nest_params = bad_layer
    with pytest.raises(ParameterError):
        return constructor('', params, nest_params)


UNIT_CHANGES = [
    {
        'change_type': 'multiplicative',
        'population': None,
        'params': {
            'V_m': 1.10,
        },
    },
    {
        'change_type': 'constant',
        'population': 'iaf_psc_alpha',
        'params': {
            'V_m': 0.0,
            'E_L': 0.0,
        },
    },
]


@pytest.fixture(params=UNIT_CHANGES)
def unit_changes(request):
    yield request.param


def test_unit_changes(base_layer, unit_changes):
    nest.ResetKernel()
    base_layer.create()
    unit_changes = dict(unit_changes)
    params = unit_changes.pop('params')
    population = unit_changes['population']
    # Save current values
    current_values = {}
    for key in params:
        current_values[key] = {
            gid: nest.GetStatus((gid,), key)[0]
            for gid in base_layer.gids(population=population)
        }
    # Apply changes
    base_layer.change_unit_states(params, **unit_changes)
    # Compare new with former
    for key, value in params.items():
        if unit_changes['change_type'] == 'constant':
            assert all(
                [
                    nest.GetStatus(
                        (gid,), key
                    )[0] == value
                    for gid in base_layer.gids(population=population)
                ]
            )
        elif unit_changes['change_type'] == 'multiplicative':
            assert all(
                [
                    nest.GetStatus(
                        (gid,), key
                    )[0] == value * current_values[key][gid]
                    for gid in base_layer.gids(population=population)
                ]
            )
        else:
            assert 0
