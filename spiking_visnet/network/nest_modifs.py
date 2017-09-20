#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nest_modifs.py

"""Gather and modify NEST and network parameters."""

import random

import nest
import numpy as np
from tqdm import tqdm


def change_synapse_states(synapse_changes):
    """Change synapse status by synapse models.

    Args:
        network (Network object): Initialized network
        synapse_changes (list): List of dictionaries each of the form::
                {synapse_model: <synapse_model>,
                 params: {<key>: <value>,
                          ...}}
            where the params contains the parameters to set for all synapses of
            a given model.

    """
    for changes in tqdm(sorted(synapse_changes, key=synapse_sorting_map),
                        desc="-> Change synapses's state."):
        nest.SetStatus(
            nest.GetConnections(synapse_model=changes['synapse_model']),
            changes['params']
            )


def synapse_sorting_map(synapse_change):
    """Map by (synapse_model, params_items) for sorting."""
    return (synapse_change['synapse_model'],
            sorted(synapse_change['params'].items()))
