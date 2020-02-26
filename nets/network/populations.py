#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/populations.py
"""Provides the ``Population`` classes."""

from pprint import pformat

from .nest_object import NestObject
from .recorders import PopulationRecorder
from .utils import if_created, if_not_created

# pylint: disable=missing-docstring


class Population(NestObject):
    """Represent a population.

    A population is defined by a (`layer_name`, `population_name`) tuple and
    contains a list of PopulationRecorder objects.
    """

    # def __init__(self, pop_name, layer_name, gids, locations, params):
    def __init__(self, name, layer, params):
        super().__init__(name, params)
        self.layer = layer
        self.params = params
        self.recorders = [
            PopulationRecorder(recorder_type, recorder_params)
            for recorder_type, recorder_params in params.get('recorders', {})
            .items()
        ]
        # Number of units per layer position
        self.number = int(self.layer.params['populations'][self.name])
        # 3D location by gid mapping
        self._locations = None
        self._created = False

    def __repr__(self):
        return '{classname}(({layer}, {population}), {params})'.format(
            classname=type(self).__name__,
            layer=self.layer.name,
            population=self.name,
            params=pformat(self.params))

    def __lt__(self, other):
        return (self.layer.name, self.name) < (other.layer.name, other.name)

    @if_not_created
    def create(self):
        if self.recorders:
            # Get all the population-wide parameters necessary for recorder
            # metadata
            # Get all gids of population
            gids = self.layer.gids(population=self.name)
            # Get locations of each gids as a (row, number, unit) tuple
            self._locations = {}
            for location in self.layer:
                location_gids = self.layer.gids(
                    population=self.name, location=location)
                for unit, gid in enumerate(location_gids):
                    self._locations[gid] = location + (unit, )
            population_params = {
                "gids": gids,
                "locations": self._locations,
                "population_name": self.name,
                "layer_name": self.layer.name,
                "layer_shape": self.layer.shape,
                "units_number": self.number,
            }

            # Create the recorders and pass along the metadata parameters
            for recorder in self.recorders:
                # Pass all the layer-wide and population-wide attributes to the
                # recorder object that deals with creating the recorder metadata
                recorder.create(population_params)

    @property
    @if_created
    def locations(self):
        return self._locations

    def get_recorders(self, recorder_type=None):
        for recorder in self.recorders:
            if recorder_type is None or recorder.type == recorder_type:
                yield recorder
