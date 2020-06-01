#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_connections.py

"""Test ``ConnectionModel`` and ``Connection`` classes."""


import nest
import nest.topology as tp

from denest.network.connections import ConnectionModel, TopoConnection


def test_full_base_layer_auto_connection(base_layer):
    nest.ResetKernel()
    base_layer.create()
    model = ConnectionModel(
        "connmodel",
        {},
        {
            "synapse_model": "static_synapse",
            "kernel": 1.0,
            "connection_type": "divergent",
        },
    )
    connection = TopoConnection(model, base_layer, None, base_layer, None)
    connection.create()
    all_gids = base_layer.gids()
    for gid in all_gids:
        # All gids are targets
        assert set(tp.GetTargetNodes((gid,), base_layer.gid)[0]) == set(all_gids)


def test_full_population_auto_connection(base_layer):
    nest.ResetKernel()
    base_layer.create()
    model = ConnectionModel(
        "connmodel",
        {},
        {
            "synapse_model": "static_synapse",
            "kernel": 1.0,
            "connection_type": "divergent",
        },
    )
    population = list(base_layer.populations.keys())[0]
    connection = TopoConnection(model, base_layer, population, base_layer, population,)
    connection.create()
    all_gids = base_layer.gids()
    pop_gids = base_layer.gids(population=population)
    for gid in all_gids:
        if gid in pop_gids:
            # All pop gids are targets
            assert set(tp.GetTargetNodes((gid,), base_layer.gid)[0]) == set(pop_gids)
        else:
            assert not tp.GetTargetNodes((gid,), base_layer.gid)[0]
