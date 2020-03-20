#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/layers.py

"""Layer objects."""

import itertools
import random

import numpy as np

from ..utils import spike_times
from .nest_object import NestObject
from .utils import flatten, if_created, if_not_created

# pylint:disable=missing-docstring


class AbstractLayer(NestObject):
    """Abstract base class for a layer.

    Defines the layer interface.
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(self, name, params):
        super().__init__(name, params)
        self._gid = None
        self._gids = None  # list of layer GIDs
        self._locations = {}  # {<gid>: (row, col)}
        self._populations = params['populations']  # {<population>: <number>}
        self.shape = params['nrows'], params['ncols']
        self.extent = params['extent']
        self.edge_wrap = params['edge_wrap']
        # Record if we change some of the layer units' state probabilistically
        self._prob_changed = False

    def __iter__(self):
        yield from itertools.product(range(self.shape[0]),
                                     range(self.shape[1]))

    @staticmethod
    def to_extent_units(value, extent, rows, columns):
        """Convert a value from grid units to extent units."""
        size = max(rows, columns)
        units = extent / size
        return value * units

    def extent_units(self, value):
        """Convert a value from grid units to extent units."""
        raise NotImplementedError

    @if_not_created
    def create(self):
        """Create the layer in NEST."""
        raise NotImplementedError

    @property
    def populations(self):
        """Return ``{<population_name>: <number_units>}`` dictionary."""
        return self._populations

    @property
    @if_created
    def gid(self):
        """Return the NEST global ID (GID) of the layer."""
        return self._gid

    @if_created
    def connect(self, target, nest_params):
        """Connect to target layer. Called by `Connection.create()`"""
        # NOTE: Don't use this method directly; use a Connection instead
        from nest import topology as tp
        tp.ConnectLayers(self.gid, target.gid, nest_params)

    def gids(self, population=None, location=None):
        """Return element GIDs, optionally filtered by population/location.

        Args:
            population (str or Sequence(str)): Matches any population name that
                has ``population`` as a substring.
            location (tuple[int] or Sequence[tuple[int]]): The location(s) to
                filter by; can be a single coordinate pair or a sequence of
                coordinate pairs.

        Returns:
            list: The GID(s).
        """
        raise NotImplementedError

    def element(self, *args):
        """Return the element(s) at the given location(s).

        Args:
            *args (tuple[int]): Coordinate pair(s) of grid location(s).

        Returns:
            tuple[tuple[int, str-like]]: For each (x, y) coordinate pair in
            ``args``, returns a tuple of (GID, population) pairs for the
            elements at that location.
        """
        raise NotImplementedError

    def change_unit_states(self, changes_dict, population=None, proportion=1.0,
                           change_type='constant'):
        """Set parameters for some units.
        
        Args:
            changes_dict (dict): Dictionary specifying changes applied to
                selected units, of the form::
                    {
                        <param_1>: <change_value_1>,
                        <param_2>: <change_value_2>
                    }
                The values are set multiplicatively or without modification
                depending on the ``change_type`` parameter.
            population (str | None): Name of population from which we select
                units. All layer's populations if None.
            proportion (float): Proportion of candidate units to which the
                changes are applied. (default 1.0)
            change_type (str): 'constant' (default) or 'multiplicative'.
                Specifies how parameter values are set from ``change_dict``. If
                "constant", the values in ``change_dict`` are set to the
                corresponding parameters without modification. If
                "multiplicative", the values in ``change_dict`` are multiplied
                to the current value of the corresponding parameter for each
                unit.
        """
        if change_type not in ['constant', 'multiplicative']:
            raise ValueError(
                "``change_type`` argument should 'constant' or 'multiplicative'"
            )
        if proportion > 1 or proportion < 0:
            raise ValueError('``proportion`` parameter should be within [0, 1]')
        if not changes_dict:
            return
        if self._prob_changed and proportion != 1.0:
            raise Exception("Attempting to change probabilistically some "
                            "units' state multiple times.")
        all_gids = self.gids(population=population)
        if proportion != 1.0:
            print(f'----> Select subset of gids (proportion = {proportion})')
        gids_to_change = self.get_gids_subset(
            all_gids,
            proportion
        )
        print(f'----> Apply "{change_type}" parameter changes on '
              f'{len(gids_to_change)}/{len(all_gids)} units '
              f'(layer={self.name}, population={population})')
        self.apply_unit_changes(gids_to_change, changes_dict,
                                change_type=change_type)
        self._prob_changed = True

    @staticmethod
    def apply_unit_changes(gids_to_change, changes_dict,
                           change_type='constant'):
        """Change the state of a list of units."""
        assert change_type in ['constant', 'multiplicative']
        import nest
        if change_type == 'constant':
            nest.SetStatus(gids_to_change, changes_dict)
        elif change_type == 'multiplicative':
            for gid, (change_key, change_ratio) in itertools.product(
                gids_to_change,
                changes_dict.items()
            ):
                current_value = nest.GetStatus((gid,), change_key)[0]
                nest.SetStatus(
                    (gid,),
                    {change_key: current_value * change_ratio}
                )

    @staticmethod
    def get_gids_subset(gids_list, proportion):
        """Return a proportion of gids picked randomly from a list."""
        return [gids_list[i] for i
                in sorted(
                    random.sample(
                        range(len(gids_list)),
                        int(len(gids_list) * proportion))
                        )]

class Layer(AbstractLayer):

    def __init__(self, name, params):
        super().__init__(name, params)
        # TODO: use same names
        self.nest_params = {
            'rows': self.shape[0],
            'columns': self.shape[1],
            'extent': self.extent,
            'edge_wrap': self.edge_wrap,
            'elements': self.build_elements(),
        }

    def extent_units(self, value):
        return self.to_extent_units(value, self.extent[0],
                                    self.shape[0], self.shape[1])

    def build_elements(self):
        """Convert ``populations`` parameters to format expected by NEST

        From the ``populations`` layer parameter, which is a dict of the
        form::
            {
                <population_name>: <number_of_units>
            }
        return a NEST element specification, which is a list of the form::
            [<model_name>, <number of units>]
        """
        populations = self.params['populations']
        if (
            not populations 
            or any([not isinstance(n, int) for n in populations.values()])
        ):
            raise ValueError(
                "Invalid format for `populations` parameter {populations} of "
                f"layer {str(self)}: expects non-empty dictionary of the form"
                "`{<population_name>: <number_of_units>}` with integer values"
            )
        # Map types to numbers
        return flatten([population, number]
                       for population, number in populations.items())

    @if_not_created
    def create(self):
        """Create the layer in NEST and update attributes."""
        import nest
        from nest import topology as tp
        self._gid = tp.CreateLayer(self.nest_params)
        # Update _locations: ``{gid: (row, col)}``
        for i, j in itertools.product(range(self.shape[0]),
                                      range(self.shape[1])):
            # IMPORTANT: rows and columns are switched in the GetElement query
            gids = tp.GetElement(self.gid, locations=(j, i))
            for gid in gids:
                self._locations[gid] = (i, j)
        # Get all GIDs
        self._gids = tuple(sorted(self._locations.keys()))

    @if_created
    def gids(self, population=None, location=None):
        """Return layer GIDs filtered by population or location."""
        import nest
        return [
            gid for gid in self._gids
            if (
                (population is None
                 or nest.GetStatus((gid,), 'model')[0] == population)
                and (location is None
                     or self.locations[gid] == location)
            )
        ]

    @if_created
    def element(self, *args):
        return tuple(self._elements[location] for location in args)

    @property
    @if_created
    def locations(self):
        """Return ``{<gid>: (<row>, <col>)}`` dict of locations."""
        return self._locations

    @if_created
    @staticmethod
    def position(*args):
        import nest.topology as tp
        return tp.GetPosition(args)

    def population_names(self):
        """Return a list of population names within this layer."""
        return list(self.params['populations'].keys())

    def recordable_population_names(self):
        """Return list of names of recordable population names in this layer."""
        return self.population_names()

    @if_created
    def find_center_element(self, population=None):
        """Return GID of an element centered within the layer."""
        center_loc = (int(self.shape[0]/2),
                      int(self.shape[1]/2))
        center_gid = self.gids(location=center_loc, population=population)[0:1]
        assert len(center_gid) == 1
        return center_gid

    @if_created
    def set_state(self, variable, values, population=None):
        """Set the state of a variable for all units in a layer.

        If value is a 2D array the same size as the layer, set the values of
        variable per location.
        """
        import nest
        if isinstance(values, np.ndarray):
            value_per_location = True
            assert np.shape(values) == self.shape, (
                "Array has the wrong shape for setting layer values"
            )
        for location in self:
            value = values[location] if value_per_location else values
            nest.SetStatus(self.gids(population=population,
                                     location=location),
                           {variable: value})


class InputLayer(Layer):
    """A layer that provides input to the network.

    A layer with a population of stimulation devices connected to an extra
    population of parrot neurons, and that can handle input arrays
    """

    PARROT_MODEL = 'parrot_neuron'
    STIMULATORS = ['spike_generator', 'poisson_generator']

    def __init__(self, name, params):
        populations = params['populations']
        if len(populations) != 1:
            raise ValueError('InputLayer must have only one population (of'
                             'stimulation devices)')
        # Save the stimulator type
        stimulator_model, nunits = list(populations.items())[0]
        if nunits != 1:
            raise ValueError(
                'InputLayer can have only one stimution device per location.'
                f'Please check the `population` parameter: {populations}'
        )
        self.stimulator_model = stimulator_model
        self.stimulator_type = None  # TODO: Check stimulator type
        # Add a parrot population entry
        populations[self.PARROT_MODEL] = 1
        params['populations'] = populations

        # Initialize the layer
        super().__init__(name, params)

    def create(self):
        """Create the layer and connect the stimulator and parrot populations"""
        super().create()
        from nest import topology as tp
        import nest
        # Connect stimulators to parrots, one-to-one
        radius = self.extent_units(0.1)
        one_to_one_connection = {
            'sources': {'model': self.stimulator_model},
            'targets': {'model': self.PARROT_MODEL},
            'connection_type': 'convergent',
            'synapse_model': 'static_synapse',
            'mask': {'circular': {'radius': radius}}
        }
        tp.ConnectLayers(self._gid, self._gid, one_to_one_connection)
        # Get stimulator type
        self.stimulator_type = nest.GetDefaults(self.stimulator_model,
                                                'type_id')
        if self.stimulator_type not in self.STIMULATORS:
            raise ValueError(
                f"The stimulation device in input layer {self.name} if not"
                f" of an accepted type. stimulator_type={self.stimulator_type},"
                f" Supported types: {self.STIMULATORS}"
            )


    def set_input(self, stimulus_array, start_time=0.):
        """Set stimulator state from stimulus_array."""
        # TODO: Remove layer `input_rate_scale_factor`
        input_rate_scale_factor = float(self.params['input_rate_scale_factor'])
        effective_max = input_rate_scale_factor * np.max(stimulus_array)
        assert stimulus_array.ndim == 3
        print(f'-> Setting input for `{self.name}`.')
        print(f'--> Rate scaling factor: {str(input_rate_scale_factor)}')
        print(f'--> Max instantaneous rate: {str(effective_max)}Hz')
        rates = input_rate_scale_factor * stimulus_array
        if self.stimulator_type == 'poisson_generator':
            print(
                f"Stimulator is a 'poisson_generator' -> Using only first frame"
                f"of the {rates.shape}-ndarray stimulus array"
            )
            # Use only first frame
            self.set_state('rate', rates[0], population=self.stimulator_model)
        elif self.stimulator_type == 'spike_generator':
            all_spike_times = spike_times.draw_spike_times(
                rates,
                start_time=start_time
            )
            self.set_state('spike_times', all_spike_times,
                           population=self.stimulator_model)
        else:
            raise NotImplementedError
    # pylint: disable=arguments-differ

    def find_center_element(self, population=None):
        return self.layers[0].find_center_element(population=population)

    def recordable_population_names(self):
        """Return list of names of recordable population names in this layer."""
        return ['parrot_neuron']
