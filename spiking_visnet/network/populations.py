#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/populations.py

"""Provides the ``Population`` and ``Recorder`` classes."""

from itertools import product
from os.path import join
from pprint import pformat

import matplotlib.pyplot as plt
import pylab

from .. import save
from ..utils import format_recorders
from .nest_object import NestObject
from .utils import if_created, if_not_created


class Population(NestObject):
    """Represent a population.

    A population is defined by a (`layer_name`, `population_name`) tuple and
    contains a list of Recorder objects.
    """
    # def __init__(self, pop_name, layer_name, gids, locations, params):
    def __init__(self, name, layer, params):
        super().__init__(name, params)
        self.layer = layer
        self.params = params
        self.recorders = [Recorder(recorder_type, recorder_params)
                          for recorder_type, recorder_params
                          in params.get('recorders', {}).items()]
        self.number = self.layer.params['populations'][self.name]
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
        # Get all gids of population
        gids = self.layer.gids(population=self.name)
        # Get locations of each gids as a (row, number, unit) tuple
        self._locations = {}
        for location in self.layer:
            location_gids = self.layer.gids(population=self.name,
                                            location=location)
            for unit, gid in enumerate(location_gids):
                self._locations[gid] = location + (unit,)
        for recorder in self.recorders:
            recorder.create(gids, self.locations)

    @property
    @if_created
    def locations(self):
        return self._locations

    def save(self, output_dir, with_rasters=True):
        if with_rasters:
            self.save_rasters(output_dir)
        self.save_recorders(output_dir)

    def save_recorders(self, output_dir):
        import nest
        ntimesteps = int(nest.GetKernelStatus('time') /
                         nest.GetKernelStatus('resolution'))
        formatted_shape = (ntimesteps,) + self.layer.shape
        for unit_index, recorder in product(range(self.number),
                                            self.recorders):
            for variable in recorder.variables:
                activity = recorder.formatted_data(
                    formatted_shape=formatted_shape,
                    variable=variable,
                    unit_index=unit_index
                )
                recorder_path = save.output_path(
                    output_dir,
                    'recorders',
                    self.layer.name,
                    self.name,
                    unit_index=unit_index,
                    variable=variable
                )
                save.save_array(recorder_path, activity)

    def save_rasters(self, output_dir):
        for recorder in self.recorders:
            raster = recorder.get_nest_raster()
            if raster is not None:
                pylab.title(self.layer.name + '_' + self.name)
                f = raster[0].figure
                f.set_size_inches(15, 9)
                f.savefig(save.output_path(output_dir, 'rasters',
                                           self.layer.name, self.name),
                          dpi=100)
                plt.close()


class Recorder(NestObject):
    """Represent a recorder node.

    Handles connecting the recorder node to the population and formatting the
    recorder's data.
    """
    def __init__(self, name, params):
        super().__init__(name, params)
        self._gids = None
        self._locations = None
        self._gid = None
        self._files = None
        self._record_to = None
        self._record_from = None
        self._type = None
        if self.name in ['multimeter', 'spike_detector']:
            self._type = self.name
        else:
            # TODO: access somehow the base nest model from which the recorder
            # model inherits.
            raise Exception('The recorder type is not recognized.')

    @if_not_created
    def create(self, gids, locations):
        import nest
        # Save gids and locations
        self._gids = gids
        self._locations = locations
        # Create node
        self._gid = nest.Create(self.name, params=self.params)
        # Get node parameters from nest (possibly nest defaults)
        self._record_to = nest.GetStatus(self.gid, 'record_to')[0]
        if self.type == 'multimeter':
            self._record_from = [str(variable) for variable
                                 in nest.GetStatus(self.gid, 'record_from')[0]]
        elif self.type == 'spike_detector':
            self._record_from = ['spikes']
        # Connect population
        if self.type == 'multimeter':
            nest.Connect(self.gid, self.gids)
        elif self.type == 'spike_detector':
            nest.Connect(self.gids, self.gid)

    @property
    @if_created
    def gid(self):
        return self._gid

    @property
    @if_created
    def gids(self):
        return self._gids

    @property
    @if_created
    def locations(self):
        return self._locations

    @property
    @if_created
    def variables(self):
        return self._record_from

    @property
    def type(self):
        return self._type

    def formatted_data(self, formatted_shape=None, variable=None,
                       unit_index=0):
        return format_recorders.format_recorder(
            self.gid,
            recorder_type=self.type,
            shape=formatted_shape,
            locations=self.locations,
            variable=variable,
            unit_index=unit_index
        )

    def get_nest_raster(self):
        import nest
        from nest import raster_plot
        if (self.type == 'spike_detector'
                and 'memory' in self._record_to
                and len(nest.GetStatus(self.gid)[0]['events']['senders'])):
            return raster_plot.from_device(self.gid, hist=True)
        return None
