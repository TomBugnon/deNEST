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
        self.recorders = [
            Recorder(recorder_type, recorder_params)
            for recorder_type, recorder_params in params.get('recorders', {})
            .items()
        ]
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
            # Get all the population-wide parameters necessary for recorder
            # formatting
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

            # Create the recorders and pass along the formatting parameters
            for recorder in self.recorders:
                # Pass all the layer-wide and population-wide attributes to the
                # recorder object that deals with data formatting
                recorder.create(population_params)

    @property
    @if_created
    def locations(self):
        return self._locations

    def get_recorders(self, recorder_type=None):
        for recorder in self.recorders:
            if recorder_type is None or recorder.type == recorder_type:
                yield recorder


class Recorder(NestObject):
    """Represent a recorder node. Connects to a single population.

    Handles connecting the recorder node to the population, formatting the
    recorder's data and saving the formatted data.
    The recorder objects contains the population and layer specs necessary for
    formatting (shape, locations, etc).
    """
    def __init__(self, name, params):
        super().__init__(name, params)
        # Attributes below are taken from NEST kernel after creation
        # Population object during `create()` call
        self._gid = None # gid of recorder node
        self._files = None # files of recorded data (None if to memory)
        self._record_to = None # eg ['memory', 'file']
        self._record_from = None # list of variables for mm, or ['spikes']
        # Attributes below are necessary for formatting and are passed by the
        # Population object during `create()` call
        self._gids = None # all gids of recorded nodes
        self._locations = None # location by gid for recorded nodes
        self._population_name = None # Name of recorded (parent) population
        self._layer_name = None # Name of recorded (parent) pop's layer
        self._layer_shape = None # (nrows, cols) for recorded pop
        self._units_number = None # Number of nodes per grid position for pop
        ##
        self._type = None # 'spike detector' or 'multimeter'
        if self.name in ['multimeter', 'spike_detector']:
            self._type = self.name
        else:
            # TODO: access somehow the base nest model from which the recorder
            # model inherits.
            raise Exception('The recorder type is not recognized.')

    @if_not_created
    def create(self, population_params):
        """Get the layer/pop necessary attributes, create node and connect."""
        import nest
        # Save population and layer-wide attributes
        self._gids = population_params['gids']
        self._locations = population_params['locations']
        self._population_name = population_params['population_name']
        self._layer_name = population_params['layer_name']
        self._layer_shape = population_params['layer_shape']
        self._units_number = population_params['units_number']
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

    def save_raster(self, output_dir):
        if self.type == 'spike_detector':
            raster, error_msg = self.get_nest_raster()
            if raster is not None:
                pylab.title(self._layer_name + '_' + self._population_name)
                f = raster[0].figure
                f.set_size_inches(15, 9)
                f.savefig(save.output_path(output_dir, 'rasters',
                                           self._layer_name,
                                           self._population_name),
                          dpi=100)
                plt.close()
            else:
                print(f'Not saving raster for population:'
                      f' {str(self._population_name)}:')
                print(f'-> {error_msg}\n')

    def save(self, output_dir):
        """Save the formatted activity of recorders.

        NB: Since we load and manipulate the activity for all variables recorded
        by a recorder at once (for speed), this can get hard on memory when many
        variables are recorded. If you experience memory issues, a possiblity is
        to create separate recorders for each variable.
        """
        # Get formatted arrays for each variable and each unit_index.
        # all_recorder_activity = {'var1': [activity_unit_0,
        #                                   activity_unit_1,...]}
        all_recorder_activity = self.formatted_data()

        # Save the formatted arrays separately for each var and unit_index
        for variable, unit_index in product(self.variables,
                                            range(self._units_number)):

            recorder_path = save.output_path(
                output_dir,
                'recorders',
                self._layer_name,
                self._population_name,
                unit_index=unit_index,
                variable=variable
            )
            save.save_array(recorder_path,
                            all_recorder_activity[variable][unit_index])

    def formatted_data(self):
        # Throw a warning if the interval is below the millisecond as that won't
        # be taken in account during formatting.
        import nest

        # NB: We only sample at 1ms !
        ntimesteps = int(nest.GetKernelStatus('time'))
        formatted_shape = (ntimesteps,) + self._layer_shape
        if (self.type == 'multimeter'
            and nest.GetStatus(self.gid, 'interval')[0] < 1):
            import warnings
            warnings.warn('NB: The multimeter interval is below 1msec, but we'
                          'only format at the msec scale!')

        return format_recorders.format_recorder(
            self.gid,
            recorder_type=self.type,
            shape=formatted_shape,
            locations=self._locations,
            all_variables=self.variables,
            all_unit_indices=range(self._units_number)
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
