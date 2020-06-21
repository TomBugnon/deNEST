#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_layers.py


"""Test ``Layer`` class."""

import nest
import pytest
from pytest import approx

from denest.network.layers import InputLayer, Layer
from denest.utils.validation import ParameterError

BASE_LAYERS = [
    (Layer, {"populations": {"iaf_psc_alpha": 2}}, {"rows": 1, "columns": 1}),
    (
        Layer,
        {"populations": {"iaf_psc_alpha": 2, "iaf_cond_alpha": 2}},
        {"rows": 2, "columns": 2, "edge_wrap": True},
    ),
]

INPUT_LAYERS = [
    (InputLayer, {"add_parrots": True, "populations": {"poisson_generator": 1}}, {"rows": 2, "columns": 2}),
    (InputLayer, {"add_parrots": False, "populations": {"poisson_generator": 1}}, {"rows": 2, "columns": 2}),
]


BAD_LAYERS = [
    (Layer, {}, {"rows": 1, "columns": 1}),
    (Layer, {"populations": {"iaf_psc_alpha": 1}}, {"columns": 1}),
    (Layer, {"populations": {"iaf_psc_alpha": 1}}, {"rows": 1}),
    (
        Layer,
        {"populations": {"iaf_psc_alpha": 1}},
        {"rows": 1, "columns": 1, "elements": "iaf_psc_alpha"},
    ),
]


def init_layer(constructor, params, nest_params):
    layer = constructor("", params, nest_params)
    yield layer


def test_layer(layer):
    layer.create()
    # Correct gids
    assert set(nest.GetLeaves(layer.gid)[0]) == set(layer.gids())
    for population, pop_number in layer.populations.items():
        # Correct model
        assert all(
            [
                nest.GetStatus((gid,), "model")[0] == population
                for gid in layer.gids(population=population)
            ]
        )
        # Correct number of units at each location
        assert all(
            [
                len(layer.gids(population=population, location=location)) == pop_number
                for location in layer
            ]
        )


@pytest.fixture(params=BAD_LAYERS)
def bad_layer(request):
    yield request.param


def test_bad_layer(bad_layer):
    constructor, params, nest_params = bad_layer
    with pytest.raises(ParameterError):
        return constructor("", params, nest_params)


UNIT_CHANGES = [
    {
        "change_type": "multiplicative",
        "population_name": None,
        "nest_params": {"V_m": 1.10},
        "from_array": False,
    },
    {
        "change_type": "constant",
        "population_name": "iaf_psc_alpha",
        "nest_params": {"V_m": 0.0, "E_L": 0.0},
        "from_array": False,
    },
    {
        "change_type": "additive",
        "population_name": "iaf_psc_alpha",
        "nest_params": {"V_m": 1.0, "E_L": 1.0},
        "from_array": False,
    },
    {
        "change_type": "multiplicative",
        "population_name": "iaf_psc_alpha",
        "nest_params": {"V_m": None, "E_L": None},
        "from_array": True,
    },
    {
        "change_type": "constant",
        "population_name": "iaf_psc_alpha",
        "nest_params": {"V_m": None, "E_L": None},
        "from_array": True,
    },
    {
        "change_type": "additive",
        "population_name": "iaf_psc_alpha",
        "nest_params": {"V_m": None, "E_L": None},
        "from_array": True,
    },
]


@pytest.fixture(params=UNIT_CHANGES)
def unit_changes(request):
    yield request.param


def test_set_state(base_layer, unit_changes):
    import numpy as np

    nest.ResetKernel()
    base_layer.create()
    # Save current values
    current_values = {}
    for key in unit_changes['nest_params'].keys():
        current_values[key] = {
            gid: nest.GetStatus((gid,), key)[0]
            for gid in base_layer.gids(population=unit_changes['population_name'])
        }

    # For from_array changes, create the array on the fly because we need the
    # population shape (hacky)
    if unit_changes['from_array']:
        unit_changes['nest_params'] = {
            key: np.around(
                np.random.rand(
                    *base_layer.population_shape[unit_changes['population_name']]
                ),
                decimals=5
            )
            for key in unit_changes['nest_params'].keys()
        }

    # Apply changes
    base_layer.set_state(
        **unit_changes
    )
    # Compare new with former
    for key, value in unit_changes['nest_params'].items():

        for population in base_layer.population_names:
            if unit_changes['population_name'] is not None \
                    and unit_changes['population_name'] != population:
                continue

            if unit_changes['from_array']:
                values_array = value
            else:
                values_array = np.full(
                    base_layer.population_shape[population],
                    value,
                )

            if unit_changes["change_type"] == "constant":
                assert all(
                    [
                        nest.GetStatus((gid,), key)[0] == approx(
                            values_array[base_layer.population_locations[gid]]
                        )
                        for gid in base_layer.gids(population=population)
                    ]
                )
            elif unit_changes["change_type"] == "multiplicative":
                assert all(
                    [
                        nest.GetStatus((gid,), key)[0] == approx(
                            values_array[base_layer.population_locations[gid]] \
                            * current_values[key][gid]
                        )
                        for gid in base_layer.gids(population=population)
                    ]
                )
            elif unit_changes["change_type"] == "additive":
                assert all(
                    [
                        nest.GetStatus((gid,), key)[0] == approx(
                            values_array[base_layer.population_locations[gid]] \
                            + current_values[key][gid]
                        )
                        for gid in base_layer.gids(population=population)
                    ]
                )
            else:
                assert 0
