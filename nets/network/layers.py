#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/layers.py

"""Layer objects."""

import itertools
import random

import numpy as np
from tqdm import tqdm

from ..utils import filter_suffixes, spike_times
from .nest_object import NestObject
from .utils import flatten, if_created, if_not_created


class AbstractLayer(NestObject):
    """Abstract base class for a layer.

    Defines the layer interface.
    """

    def __init__(self, name, params):
        super().__init__(name, params)
        self._gid = None
        self._gids = None
        self._elements = None
        self._locations = None
        self._populations = None
        self.shape = params['nrows'], params['ncols']
        self.extent = (params['visSize'],) * 2
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

    def create(self):
        """Create the layer in NEST."""
        raise NotImplementedError

    @property
    def populations(self):
        return self._populations

    @property
    @if_created
    def gid(self):
        """Return the NEST global ID (GID) of the layer."""
        return self._gid

    @if_created
    def connect(self, target, nest_params):
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

    def location(self, *args):
        """Return the location(s) on the layer grid of the GID(s).

        Args:
            *args (int): The GID(s) of interest.

        Returns:
            tuple[tuple[int]]: Returns a tuple of (x, y) coordinate pairs, one
            for each GID in ``gids``, giving the location of the element with
            that GID.
        """
        raise NotImplementedError

    def population(self, *args):
        """Return the population(s) of the GID(s).

        Args:
            *args (int): The GID(s) of interest.

        Returns:
            tuple[str]: Returns a tuple the population names of the GID(s).
        """
        raise NotImplementedError

    def change_unit_states(self, changes_dict, population=None, proportion=1.0):
        """Call nest.SetStatus for a proportion of units."""
        if not changes_dict:
            return
        if self._prob_changed and proportion != 1.0:
            raise Exception("Attempting to change probabilistically some "
                            "units' state multiple times.")
            return
        gids_to_change = self.get_gids_subset(
            self.gids(population=population),
            proportion
        )
        self.apply_unit_changes(gids_to_change, changes_dict)
        self._prob_changed = True

    @staticmethod
    def apply_unit_changes(gids_to_change, changes_dict):
        """Change the state of a list of units."""
        import nest
        nest.SetStatus(gids_to_change, changes_dict)

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
            'rows': self.params['nrows'],
            'columns': self.params['ncols'],
            'extent': [self.params['visSize']] * 2,
            'edge_wrap': self.params.get('edge_wrap', False),
            'elements': self.build_elements(),
        }
        # TODO: implement
        self._connected = list()
        self._gid = None

    def extent_units(self, value):
        return self.to_extent_units(value, self.extent[0],
                                    self.shape[0], self.shape[1])

    def build_elements(self):
        """Return the NEST description of layer elements.

        A NEST element specification is a list of the form::

            [<model_name>, <model_number>, <model_name>, <model_number>, ...]

        This converts the parameters to such a list.
        """
        populations = self.params['populations']
        assert populations
        # Map types to numbers
        return flatten([population, number]
                       for population, number in populations.items())

    @if_not_created
    def create(self):
        import nest
        from nest import topology as tp
        self._gid = tp.CreateLayer(self.nest_params)
        # Maps grid location to elements ((GID, population) pair)
        self._elements = dict()
        # Maps GID to location
        self._locations = dict()
        # Maps GID to population
        self._populations = dict()
        for i, j in itertools.product(range(self.shape[0]),
                                      range(self.shape[1])):
            # IMPORTANT: rows and columns are switched in the GetElement query
            gids = tp.GetElement(self.gid, locations=(j, i))
            populations = [
                str(model) for model in nest.GetStatus(gids, 'model')
            ]
            elements = tuple(zip(gids, populations))
            self._elements[(i, j)] = elements
            for gid, population in elements:
                self._locations[gid] = (i, j)
                self._populations[gid] = population
        # Get all GIDs
        self._gids = tuple(sorted(self._locations.keys()))

    @if_created
    def gids(self, population=None, location=None):
        pop_filt = None
        if population is not None:
            def pop_filt(gid):
                return population in self._populations[gid]
        loc_filt = None
        if location is not None:
            def loc_filt(gid):
                return (self._locations[gid] == location or
                        self._locations[gid] in location)
        return sorted(tuple(filter(loc_filt, filter(pop_filt, self._gids))))

    @if_created
    def element(self, *args):
        return tuple(self._elements[location] for location in args)

    @if_created
    def location(self, *args):
        return tuple(self._locations[gid] for gid in args)

    @if_created
    def position(self, *args):
        import nest.topology as tp
        return tp.GetPosition(args)

    @if_created
    def population(self, *args):
        return tuple(self._populations[gid] for gid in args)

    def population_names(self):
        """Return a list of population names within this layer."""
        return list(self.params['populations'].keys())

    @if_created
    def find_center_element(self, population=None):
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
        value_per_location = (isinstance(values, np.ndarray)
                              and np.shape(values) == self.shape)
        for location in self:
            value = values[location] if value_per_location else values
            nest.SetStatus(self.gids(population=population,
                                     location=location),
                           {variable: value})


class InputLayer(AbstractLayer):
    """A layer that provides input to the network.

    This layer consists of several sublayers, each distinct NEST topological
    layers with their own GID, represented by a ``Layer``:

      - For each filter combination, a stimulator/parrot pair of layers is
        created.
      - In such a pair, the stimulator layer contains stimulator devices, while
        the parrot layer passes the stimuli from the stimulator layer to
        multiple outputs.
    """

    PARROT_MODEL = 'parrot_neuron'

    def __init__(self, name, params):
        super().__init__(name, params)
        # Add parrot populations
        # ~~~~~~~~~~~~~~~~~~~~~~
        populations = self.params['populations']
        # Check that there's only one stimulator type
        if len(populations) != 1:
            raise ValueError('InputLayer must have only one population')
        # Save the the stimulator type and get its number
        self.stimulator_model, number = list(populations.items())[0]
        self.stimulator_type = None
        # Add a parrot population entry
        populations[self.PARROT_MODEL] = number
        # Make a duplicate sublayer for each filter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        names = filter_suffixes.get_expanded_names(self.name,
                                                   self.params.get('filters',
                                                                   {}))
        self.layers = [Layer(name, self.params) for name in names]
        # TODO: Possibly scale the weights of all input connections by the
        # number of input layers

    def extent_units(self, value):
        # IMPORTANT: Assumes all sublayers are the same size!
        return self.layers[0].extent_units(value)

    def _layer_get(self, attr_name):
        """Get an attribute from each sublayer."""
        return tuple(getattr(layer, attr_name) for layer in self.layers)

    def _layer_call(self, method_name, *args, **kwargs):
        """Call a method on each sublayer."""
        return tuple(method(*args, **kwargs)
                     for method in self._layer_get(method_name))

    def _connect(self, target, nest_params):
        self._layer_call('_connect', target, nest_params)

    @if_not_created
    def create(self):
        from nest import topology as tp
        import nest
        # Create sublayers
        self._layer_call('create')
        # Set the GID
        self._gid = flatten(self._layer_get('gid'))
        # Connect stimulators to parrots, one-to-one
        # IMPORTANT: This assumes that all layers are the same size!
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

    @if_created
    def gids(self, population=None, location=None):
        return flatten(self._layer_call('gids',
                                        population=population,
                                        location=location))

    @if_created
    def element(self, *args):
        return flatten(self._layer_call('element', *args))

    @if_created
    def population(self, *args):
        return flatten(self._layer_call('population', *args))

    @if_created
    def location(self, *args):
        return flatten(self._layer_call('location', *args))

    def set_input(self, stimulus, start_time=0.):
        input_rate_scale_factor = float(self.params['input_rate_scale_factor'])
        effective_max = input_rate_scale_factor * np.max(stimulus['movie'])
        print(f'-> Setting input for `{self.name}`.')
        print(f'--> Rate scaling factor: {str(input_rate_scale_factor)}')
        print(f'--> Max instantaneous rate: {str(effective_max)}Hz')
        for layer in tqdm(self.layers):
            # TODO: Input layers should be able to see different filters
            layer_index = 0
            layer_rates = (input_rate_scale_factor
                           * stimulus['movie'][:, layer_index, :, :])
            if self.stimulator_type == 'poisson_generator':
                # Use only first frame
                layer.set_state('rate', layer_rates[0],
                                population=self.stimulator_model)
            elif self.stimulator_type == 'spike_generator':
                all_spike_times = spike_times.draw_spike_times(
                    layer_rates,
                    start_time=start_time
                )
                layer.set_state('spike_times', all_spike_times,
                                population=self.stimulator_model)
            else:
                raise NotImplementedError
    # pylint: disable=arguments-differ

    def find_center_element(self, population=None):
        return self.layers[0].find_center_element(population=population)
