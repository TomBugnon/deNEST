#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# recorders.py

"""Create and save population and connection recorder objects."""

# pylint:disable=missing-docstring

import os
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import pylab

from .. import save
from ..utils import format_recorders, misc
from .nest_object import NestObject
from .utils import if_created, if_not_created

POP_RECORDER_TYPES = ['multimeter', 'spike_detector']

# Parameters that are recognized and not passed as NEST parameters. These
# parameters are set in `populations.yml` rather than `recorders.yml`
NON_NEST_PARAMS = {
    'formatting_interval': None, # Effective default is multimeter's `interval`
                                 # parameter, or 1.0ms for spike detector
}

class BaseRecorder(NestObject):
    """Base class for all recorder classes. Represent nodes (not models).

    NB: The recorder models are created separately !!! All the parameters
    passed to these classes originate not from the `recorder` parameters
    (`recorders.yml`) but from the population parameters !

    TODO: Don't create the recorder models separately so we can define
    "non_nest" parameters in `recorder.yml`.
    """
    def __init__(self, name, all_params):
        # Pop off the params that shouldn't be considered as NEST parameters
        nest_params = deepcopy(dict(all_params))
        params = {}
        for non_nest_param, default in NON_NEST_PARAMS.items():
            params[non_nest_param] = nest_params.pop(non_nest_param, default)
        # We now save the params and nest_params dictionaries as attributes
        super().__init__(name, params)
        self.nest_params = nest_params
        # Attributes below may depend on NEST default and are derived after
        # creation
        self._gid = None # gid of recorder node
        self._files = None # files of recorded data (None if to memory)
        self._record_to = None # eg ['memory', 'file']
        self._record_from = None # list of variables for mm, or ['spikes']

    @property
    @if_created
    def gid(self):
        return self._gid

    @property
    @if_created
    def variables(self):
        return self._record_from

    @property
    def type(self):
        return self._type

    def clear_memory(self):
        """Clear memory and disk from recorder data.

        NB: We clear the contents of files rather than deleting it !!! Otherwise
        NEST doesn't write anymore."""
        import nest
        if 'memory' in self._record_to:
            # Clear events by setting n_events = 0
            nest.SetStatus(self.gid, {'n_events': 0})
        if 'file' in self._record_to:
            # delete the raw files
            files = nest.GetStatus(self.gid, 'filenames')[0]
            for file in files:
                misc.delete_contents(file)

    def set_status(self, params):
        """Call nest.SetStatus to set recorder params."""
        import nest
        print(f'--> Setting status for recorder {self.name}: {params}')
        nest.SetStatus(self.gid, params)


class PopulationRecorder(BaseRecorder):
    """Represent a recorder node. Connects to a single population.

    Handles connecting the recorder node to the population, formatting the
    recorder's data and saving the formatted data.
    The recorder objects contains the population and layer specs necessary for
    formatting (shape, locations, etc).
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, name, params):
        super().__init__(name, params)
        # Attributes below are necessary for formatting and are passed by the
        # Population object during `create()` call
        self._gids = None # all gids of recorded nodes
        self._locations = None # location by gid for recorded nodes
        self._population_name = None # Name of recorded (parent) population
        self._layer_name = None # Name of recorded (parent) pop's layer
        self._layer_shape = None # (nrows, cols) for recorded pop
        self._units_number = None # Number of nodes per grid position for pop
        #
        self._formatted_unit_indices = None # Index of nodes per grid position
            # for which data is formatted
        ##
        self._type = None # 'spike detector' or 'multimeter'
        if self.name in POP_RECORDER_TYPES:
            self._type = self.name
        else:
            # We try to guess the recorder type from its name
            self._type = self.guess_rec_type()
            print(f'Guessing type for recorder `{self.name}`:'
                  f' `{self._type}`')
        # Attributes below may depend on NEST default and are updated after
        # creation
        self._interval = None # Sampling interval. Only for multimeter
        self._formatting_interval = self.params['formatting_interval'] #
            #Interval between two consecutive "slices" of the formatted array.

    def guess_rec_type(self):
        """Guess recorder type from recorder name."""
        # TODO: add to doc
        # TODO: access somehow the base nest model from which the recorder
        # model inherits rather than guessing
        for rec_type in POP_RECORDER_TYPES:
            if rec_type in self.name:
                return rec_type
        raise Exception("The type of recorder {self.name} couldn't be guessed")


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
        self._formatted_unit_indices = \
            range(population_params['number_formatted'])
        # Create node
        self._gid = nest.Create(self.name, params=self.nest_params)
        # Get attributes after creation (may depend on nest defaults)
        self._record_to = nest.GetStatus(self.gid, 'record_to')[0]
        if self.type == 'multimeter':
            self._record_from = [str(variable) for variable
                                 in nest.GetStatus(self.gid, 'record_from')[0]]
            self._interval = nest.GetStatus(self.gid, 'interval')[0]
            if self._formatting_interval is None:
                self._formatting_interval = self._interval
            assert self._formatting_interval >= self._interval
        elif self.type == 'spike_detector':
            self._record_from = ['spikes']
            if self._formatting_interval is None:
                self._formatting_interval = 1.0
        # Connect population
        if self.type == 'multimeter':
            nest.Connect(self.gid, self.gids)
        elif self.type == 'spike_detector':
            nest.Connect(self.gids, self.gid)

    @property
    @if_created
    def gids(self):
        return self._gids

    @property
    @if_created
    def locations(self):
        return self._locations

    def save_raster(self, output_dir, session_name=None):
        if self.type == 'spike_detector':
            raster, error_msg = self.get_nest_raster()
            if raster is not None:
                pylab.title(self._layer_name + '_' + self._population_name)
                f = raster[0].figure
                f.set_size_inches(15, 9)
                f.savefig(save.output_path(output_dir, 'rasters',
                                           self._layer_name,
                                           self._population_name,
                                           session_name=session_name),
                          dpi=100)
                plt.close()
            else:
                print(f'Not saving raster for population:'
                      f' {str(self._population_name)}:')
                print(f'-> {error_msg}\n')

    def save(self, output_dir, session_name=None, start_time=None,
        end_time=None):
        # pylint:disable=too-many-arguments
        """Save the formatted activity of recorders.

        NB: Since we load and manipulate the activity for all variables recorded
        by a recorder at once (for speed), this can get hard on memory when many
        variables are recorded. If you experience memory issues, a possiblity is
        to create separate recorders for each variable.
        """

        # Get formatted arrays for each variable and each unit_index.
        # all_recorder_activity = {'var1': [activity_unit_0,
        #                                   activity_unit_1,...]}
        all_recorder_activity = self.formatted_data(start_time=start_time,
                                                    end_time=end_time)

        # Save the formatted arrays separately for each var and unit_index
        for variable, unit_index in product(self.variables,
                                            self._formatted_unit_indices):

            recorder_path = save.output_path(
                output_dir,
                'recorders',
                self._layer_name,
                self._population_name,
                session_name=session_name,
                unit_index=unit_index,
                variable=variable,
                formatting_interval=self._formatting_interval,
            )
            save.save_array(recorder_path,
                            all_recorder_activity[variable][unit_index])


    # TODO: Figure out a way to get all the data
    def formatted_data(self, start_time=None, end_time=None):
        """Get formatted data.

        NB: TODO: Because the last event recorded has timestamp `end_time` - 1 (where
        `end_time` is the kernel time at the end of a session), the last frame
        of formatted arrays is always filled with 0 and the data between
        `end_time` - 1 and `end_time` is lost.
        """
        # Get shape of formatted array.
        duration = end_time - start_time
        nslices = int(duration/self._formatting_interval)
        formatted_shape = (nslices,) + self._layer_shape
        if (self.type == 'multimeter'
            and self._interval != self._formatting_interval):
            import warnings
            warnings.warn(f'NB: The multimeter interval is {self._interval},'
                          f'but the formatting interval is '
                          f'{self._formatting_interval}!')

        return format_recorders.format_recorder(
            self.gid,
            recorder_type=self.type,
            shape=formatted_shape,
            locations=self._locations,
            all_variables=self.variables,
            formatted_unit_indices=self._formatted_unit_indices,
            formatting_interval=self._formatting_interval,
            start_time=start_time,
            end_time=end_time
        )

    def get_nest_raster(self):
        """Return the nest_raster plot and possibly error message."""
        import nest
        from nest import raster_plot
        assert self.type == 'spike_detector'
        raster, error_msg = None, None
        if 'memory' not in self._record_to:
            error_msg = 'Data was not saved to memory.'
        elif not len(nest.GetStatus(self.gid)[0]['events']['senders']): # pylint: disable=len-as-condition
            error_msg = 'No events were recorded.'
        elif len(nest.GetStatus(self.gid)[0]['events']['senders']) == 1:
            error_msg = 'There was only one sender'
        else:
            try:
                raster = raster_plot.from_device(self.gid, hist=True)
            except Exception as exception: # pylint: disable=broad-except
                error_msg = (f'Uncaught exception when generating raster.\n'
                             f'--> Exception message: {exception}\n'
                             f'--> Recorder status: {nest.GetStatus(self.gid)}')
        return raster, error_msg


class ConnectionRecorder(BaseRecorder):
    """Represent a weight_recorder node. Connects to synapses.

    ConnectionRecorders are connected to synapses of at most one `Connection()`
    object (that is, population-to-population projection of a certain synapse
    type)
    Handles connecting the weight_recorder node to the synapses.
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(self, name, params):
        # params in self.params, nest_params in self.nest_params
        super().__init__(name, params)
        # Attributes below are necessary for connecting and saving and are
        # passed by the # Connection object during `create()` call
        self._connection_name = None
        self._src_layer_name = None
        self._src_population_name = None
        self._src_gids = None
        self._tgt_layer_name = None
        self._tgt_population_name = None
        self._tgt_gids = None
        ##
        self._type = None # 'weight_recorder'
        if self.name in ['weight_recorder']:
            self._type = self.name
        else:
            # TODO: access somehow the base nest model from which the recorder
            # model inherits.
            raise Exception('The weight recorder type is not recognized.')

    @if_not_created
    def create(self, conn_parameters):
        """Get the Connection necessary attributes and create node.

        The synapses from the specific connection will send spikes to this
        specific weight_recorder by defining a different "nest_synapse_model"
        with the GID of this recorder as parameter."""
        import nest
        # Get population and layer-wide attributes
        self._connection_name = conn_parameters['connection_name']
        self._src_layer_name = conn_parameters['src_layer_name']
        self._src_population_name = conn_parameters['src_population_name']
        self._src_gids = conn_parameters['src_gids']
        self._tgt_layer_name = conn_parameters['tgt_layer_name']
        self._tgt_population_name = conn_parameters['tgt_population_name']
        self._tgt_gids = conn_parameters['tgt_gids']
        # Update the parameters Create node
        self._gid = nest.Create(self.name, params=self.nest_params)
        # Get node parameters from nest (possibly nest defaults)
        self._record_to = nest.GetStatus(self.gid, 'record_to')[0]

    def save(self, output_dir, session_name=None, start_time=None,
        end_time=None, clear_memory=True, with_raster=False):
        """Save unformatted weight-recorder data."""
        # pylint:disable=too-many-arguments

        data = format_recorders.gather_raw_data_connrec(self.gid,
                                                        start_time=start_time,
                                                        end_time=end_time)
        recorder_path = save.output_path(
            output_dir,
            'connectionrecorders',
            self._connection_name,
            session_name=session_name,
        )

        save.save_dict(recorder_path, data)

        if with_raster:
            # TODO: I keep this kwarg only so that Recorder.save and
            # ConnectionRecorder.save have the same signature and can be called
            # in the same Parallel loop. There should be a better way of doing
            # this.
            pass

        if clear_memory:
            self.clear_memory()
