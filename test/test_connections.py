#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_projections.py

"""Test ``ProjectionModel`` and ``Projection`` classes."""


import nest
import nest.topology as tp

from denest.network.projections import ProjectionModel, TopoProjection


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
