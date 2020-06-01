#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_models.py

"""Test NEST neuron, simulator, recorder and synapse models ."""

import nest
import pytest

from denest.network.models import Model, SynapseModel
from denest.utils.validation import ParameterError

MODEL_PARAMS = [
    (
        Model,
        "iaf_cond_exp",
        {"nest_model": "iaf_cond_exp"},
        {"V_m": -80.0, "E_L": 30.0},
    ),  # nrn1
    (
        Model,
        "my_iaf_cond_exp",
        {"nest_model": "iaf_cond_exp"},
        {"V_m": -80.0, "E_L": 30.0},
    ),  # nrn2
    (
        Model,
        "ht_neuron",
        {"nest_model": "ht_neuron"},
        {"V_m": -80.0, "g_peak_AMPA": 30.0},
    ),  # nrn3
    (
        Model,
        "my_ht_neuron",
        {"nest_model": "ht_neuron"},
        {"V_m": -80.0, "g_peak_AMPA": 30.0},
    ),  # nrn4
    (
        Model,
        "multimeter",
        {"nest_model": "multimeter"},
        {"record_from": ("V_m",), "record_to": ("memory",), "start": 100.0},
    ),
    (
        Model,
        "my_multimeter",
        {"nest_model": "multimeter"},
        {"record_from": ("V_m",), "record_to": ("memory",)},
    ),
    (
        SynapseModel,
        "my_static_synapse_1",
        {"nest_model": "static_synapse"},
        {"delay": 10.0},
    ),
    (
        SynapseModel,
        "my_static_synapse_2",
        {
            "nest_model": "static_synapse",
            "target_neuron": "ht_neuron",
            "receptor_type": "AMPA",
        },
        {},
    ),
]


MODEL_IDS = [
    "nrn1",
    "nrn2",
    "nrn3",
    "nrn4",
    "recorder1",
    "recorder2",
    "synapse1",
    "synapse2",
]


BAD_MODEL_PARAMS = [
    (Model, "iaf_cond_exp", {}, {},),
    (
        Model,
        "af_cond_exp",
        {"UNRECOGNIZED_PARAM": None, "nest_model": "iaf_cond_exp"},
        {},
    ),
    (
        SynapseModel,
        "static_synapse",
        {"nest_model": "static_synapse", "receptor": "AMPA"},
        {},
    ),
    (
        SynapseModel,
        "static_synapse",
        {"nest_model": "static_synapse", "target_neuron": "ht_neuron"},
        {},
    ),
]


@pytest.fixture(params=MODEL_PARAMS, ids=MODEL_IDS)
def model(request):
    constructor, name, params, nest_params = request.param
    print(f"name = {name}, params={params}, nest_params={nest_params}")
    return constructor(name, params, nest_params)


def test_model(model):
    nest.ResetKernel()
    # Create model in NEST
    model.create()
    # Check that created model have correct params
    print(nest.GetDefaults(model.name))
    for key, value in model.nest_params.items():
        assert value == nest.GetDefaults(model.name, key)


@pytest.fixture(params=BAD_MODEL_PARAMS)
def bad_model(request):
    constructor, name, params, nest_params = request.param
    print(f"name = {name}, params={params}, nest_params={nest_params}")
    with pytest.raises(ParameterError):
        return constructor(name, params, nest_params)


def test_bad_model(bad_model):
    pass
