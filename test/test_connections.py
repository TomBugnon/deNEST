#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_projections.py

"""Test ``ProjectionModel`` and ``Projection`` classes."""


import nest
import nest.topology as tp
import pytest

from denest.network.projections import (
    MultiSynapseProjection, ProjectionModel, TopoProjection,
    TopoProjectionModel, MultiSynapseProjectionModel
)
from denest.network.models import SynapseModel


def test_full_base_layer_auto_projection(base_layer):
    nest.ResetKernel()
    base_layer.create()
    model = ProjectionModel(
        "connmodel",
        {},
        {
            "synapse_model": "static_synapse",
            "kernel": 1.0,
            "connection_type": "divergent",
        },
    )
    projection = TopoProjection(model, base_layer, None, base_layer, None)
    projection.create()
    all_gids = base_layer.gids()
    for gid in all_gids:
        # All gids are targets
        assert set(tp.GetTargetNodes((gid,), base_layer.gid)[0]) == set(all_gids)


def test_full_population_auto_projection(base_layer):
    nest.ResetKernel()
    base_layer.create()
    model = ProjectionModel(
        "connmodel",
        {},
        {
            "synapse_model": "static_synapse",
            "kernel": 1.0,
            "connection_type": "divergent",
        },
    )
    population = list(base_layer.populations.keys())[0]
    projection = TopoProjection(model, base_layer, population, base_layer, population,)
    projection.create()
    all_gids = base_layer.gids()
    pop_gids = base_layer.gids(population=population)
    for gid in all_gids:
        if gid in pop_gids:
            # All pop gids are targets
            assert set(tp.GetTargetNodes((gid,), base_layer.gid)[0]) == set(pop_gids)
        else:
            assert not tp.GetTargetNodes((gid,), base_layer.gid)[0]


@pytest.fixture
def source_layer(base_layer):
    return base_layer

@pytest.fixture
def target_layer(base_layer):
    return base_layer


def test_multisynapse_projection(source_layer, target_layer):
    nest.ResetKernel()
    source_layer.create()
    target_layer.create()

    # Create reference (labelled) synapse model
    base_synapse_model = SynapseModel(
        "base_synapse_model",
        {
            'nest_model': 'static_synapse_lbl',
        },
        {
            'synapse_label': 1,
        }
    )
    base_synapse_model.create()
    multisyn_syn_model = SynapseModel(
        "multisyn_synapse_model",
        {
            'nest_model': 'static_synapse_lbl',
        },
        {
            'synapse_label': 2,
        }
    )
    multisyn_syn_model.create()

    # Use reference synapse model in reference proj
    base_model = TopoProjectionModel(
        "base_model",
        {},
        {
            "synapse_model": "base_synapse_model",
            "kernel": 0.5,
            "connection_type": "divergent",
        },
    )
    base_projection = TopoProjection(
        base_model, source_layer, None, target_layer, None
    )
    base_projection.create()

    # Create multisynapse proj by mimicking label
    # Also match src/tgt population
    multisynapse_model = MultiSynapseProjectionModel(
        "multisynapse_model",
        {
            'make_symmetric': True,
            'query_synapse_label': 1,
        },
        {
            "model": "multisyn_synapse_model"
        },
    )
    src_pops = list(source_layer.populations.keys())
    src_pop = src_pops[0]
    tgt_pops = list(target_layer.populations.keys())
    tgt_pop = tgt_pops[-1]
    multi_projection = MultiSynapseProjection(
        multisynapse_model, source_layer, src_pop, target_layer, tgt_pop
    )
    multi_projection.create()

    # Compare targets for all gids, and check synapses are symmetric
    src_pop_gids = source_layer.gids(population=src_pop)
    tgt_pop_gids = target_layer.gids(population=tgt_pop)
    for gid in src_pop_gids:
        base_proj_targets_all = tp.GetTargetNodes(
                (gid,),
                target_layer.gid,
                syn_model=base_synapse_model,
        )[0]
        base_proj_targets_pop = [
            gid for gid in base_proj_targets_all
            if gid in tgt_pop_gids
        ]
        multisyn_proj_targets = set(
            tp.GetTargetNodes(
                (gid,),
                target_layer.gid,
                syn_model=multisyn_syn_model,
            )[0]
        )
        assert set(base_proj_targets_pop) == multisyn_proj_targets
        assert set(base_proj_targets_all).issuperset(multisyn_proj_targets)

        # Check there's one fb synapse
        for tgt_gid in multisyn_proj_targets:
            feedback_targets = tp.GetTargetNodes(
                (tgt_gid,),
                source_layer.gid,
                syn_model=multisyn_syn_model,
            )[0]
            assert gid in feedback_targets

    # Check number of multisynapses
    base_conns = nest.GetConnections(
        synapse_label=1,
        source=src_pop_gids,
        target=tgt_pop_gids,
    )
    multisyn_conns = nest.GetConnections(synapse_label=2)
    assert len(multisyn_conns) == 2*len(base_conns)
