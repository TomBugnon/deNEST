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
        if self.recorders:
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
        """Save the formatted activity of recorders.

        NB: Since we load and manipulate the activity for all variables recorded
        by a recorder at once (for speed), this can get hard on memory when many
        variables are recorded. If you experience memory issues, a possiblity is
        to create separate recorders for each variable.
        """
        import nest
        # NB: We only sample at 1ms !
        ntimesteps = int(nest.GetKernelStatus('time'))
        formatted_shape = (ntimesteps,) + self.layer.shape
        for recorder in self.recorders:

            all_unit_indices = range(self.number)
            all_variables = recorder.variables

            # Get formatted arrays for each variable and each unit_index.
            # all_recorder_activity = {'var1': [activity_unit_0,
            #                                  activity_unit_1,...]}
            all_recorder_activity = recorder.formatted_data(
                formatted_shape=formatted_shape,
                all_variables=all_variables,
                all_unit_indices=all_unit_indices
            )

            # Save the formatted arrays separately for each var and unit_index
            for variable, unit_index in product(all_variables,
                                                all_unit_indices):

                recorder_path = save.output_path(
                    output_dir,
                    'recorders',
                    self.layer.name,
                    self.name,
                    unit_index=unit_index,
                    variable=variable
                )
                save.save_array(recorder_path,
                                all_recorder_activity[variable][unit_index])


    def save_rasters(self, output_dir):
        for recorder in [rec for rec in self.recorders
                         if rec.type == 'spike_detector']:
            raster, error_msg = recorder.get_nest_raster()
            if raster is not None:
                pylab.title(self.layer.name + '_' + self.name)
                f = raster[0].figure
                f.set_size_inches(15, 9)
                f.savefig(save.output_path(output_dir, 'rasters',
                                           self.layer.name, self.name),
                          dpi=100)
                plt.close()
            else:
                print(f'Not saving raster for population {str(self)}:')
                print(f'-> {error_msg}\n')



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

    def formatted_data(self, formatted_shape=None, all_variables=('V_m',),
                       all_unit_indices=(0,)):
        # Throw a warning if the interval is below the millisecond as that won't
        # be taken in account during formatting.
        import nest
        if (self.type == 'multimeter'
            and nest.GetStatus(self.gid, 'interval')[0] < 1):
            import warnings
            warnings.warn('NB: The multimeter interval is below 1msec, but we'
                          'only format at the msec scale!')
        return format_recorders.format_recorder(
            self.gid,
            recorder_type=self.type,
            shape=formatted_shape,
            locations=self.locations,
            all_variables=all_variables,
            all_unit_indices=all_unit_indices
        )

    def get_nest_raster(self):
        """Return the nest_raster plot and possibly error message."""
        import nest
        from nest import raster_plot
        assert (self.type == 'spike_detector')
        raster, error_msg = None, None
        if 'memory' not in self._record_to:
            error_msg = 'Data was not saved to memory.'
        elif not len(nest.GetStatus(self.gid)[0]['events']['senders']):
            error_msg = 'No events were recorded.'
        elif len(nest.GetStatus(self.gid)[0]['events']['senders']) == 1:
            error_msg = 'There was only one sender'
        else:
            try:
                raster = raster_plot.from_device(self.gid, hist=True)
            except Exception as e:
                error_msg = (f'Uncaught exception when generating raster.\n'
                             f'--> Exception message: {e}\n'
                             f'--> Recorder status: {nest.GetStatus(self.gid)}')
        return raster, error_msg
