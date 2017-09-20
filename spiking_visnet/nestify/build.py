#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/build.py

"""Get dependent parameters from independent network parameters."""

# pylint: disable=too-few-public-methods

import functools
import itertools
import logging
import logging.config
import random
from collections import ChainMap
from copy import deepcopy
from os.path import join
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pylab

from tqdm import tqdm

from .. import save
from ..utils import filter_suffixes, format_recorders, spike_times

log = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'stdout': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'loggers': {
        'spiking_visnet': {
            'level': 'INFO',
            'handlers': ['stdout'],
        }
    }
})


def flatten(seq):
    """Flatten an iterable of iterables into a tuple."""
    return tuple(item for subseq in seq for item in subseq)


def indent(string, amount=2):
    """Indent a string by an amount."""
    return '\n'.join((' ' * amount) + line for line in string.split('\n'))


class NotCreatedError(AttributeError):
    """Raised when a ``NestObject`` needs to have been created, but wasn't."""
    pass


# pylint: disable=protected-access

def if_not_created(method):
    """Only call a method if the ``_created`` attribute isn't set.

    After calling, sets ``_created = True``.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-docstring
        if self._created:
            log.warning('Attempted to create object more than once:\n%s',
                        indent(str(self)))
            return
        try:
            self._created = True
            value = method(self, *args, **kwargs)
        except Exception as error:
            self._created = False
            raise error
        return value
    return wrapper


def if_created(method):
    """Raise an error if the `_created` attribute is not set."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-docstring
        if not self._created:
            raise NotCreatedError('Must call `create()` first:\n' +
                                  indent(str(self)))
        return method(self, *args, **kwargs)
    return wrapper

# pylint: enable=protected-access


@functools.total_ordering
class NestObject:
    """Base class for a named NEST object.

    Args:
        name (str): The name of the object.
        params (Params): The object parameters.

    Objects are ordered and hashed by name.
    """

    def __init__(self, name, params):
        self.name = name
        # Flatten the parameters to a dictionary (and make a copy)
        self.params = dict(params)
        # Whether the object has been created in NEST
        self._created = False

    # pylint: disable=unused-argument,invalid-name
    def _repr_pretty_(self, p, cycle):
        opener = '{classname}({name}, '.format(
            classname=type(self).__name__, name=self.name)
        closer = ')'
        with p.group(p.indentation, opener, closer):
            p.breakable()
            p.pretty(self.params)
    # pylint: enable=unused-argument,invalid-name

    def __repr__(self):
        return '{classname}({name}, {params})'.format(
            classname=type(self).__name__,
            name=self.name,
            params=pformat(self.params))

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def __getattr__(self, name):
        try:
            return self.params[name]
        except KeyError:
            return self.__getattribute__(name)


class Model(NestObject):
    """Represent a model in NEST."""

    def __init__(self, name, params):
        super().__init__(name, params)
        # Save and remove the NEST model name from the nest parameters.
        self.nest_model = self.params.pop('nest_model')
        # TODO: keep nest params in params['nest_params'] and leave base model
        # as params['nest_model']?
        self.nest_params = dict(self.params)

    @if_not_created
    def create(self):
        """Create or update the NEST model represented by this object.

        If the name of the base nest model and of the model to be created are
        the same, update (change defaults) rather than copy the base nest
        model.
        """
        import nest
        if not self.nest_model == self.name:
            nest.CopyModel(self.nest_model, self.name, self.nest_params)
        else:
            nest.SetDefaults(self.nest_model, self.nest_params)

class SynapseModel(Model):
    """Represents a NEST synapse.

    ..note::
        NEST expects 'receptor_type' to be an integer rather than a string. The
        integer index must be found in the defaults of the target neuron.
    """
    def __init__(self, name, params):
        super().__init__(name, params)
        # Replace the target receptor type with its NEST index
        if 'receptor_type' in params:
            if 'target_neuron' not in params:
                raise ValueError("must specify 'target_neuron' "
                                 "if providing 'receptor_type'")
            import nest
            target = self.nest_params.pop('target_neuron')
            receptors = nest.GetDefaults(target)['receptor_types']
            self.nest_params['receptor_type'] = \
                receptors[self.params['receptor_type']]


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

    def __iter__(self):
        yield from itertools.product(range(self.shape[0]),
                                     range(self.shape[1]))

    @staticmethod
    def to_extent_units(value, extent, rows, columns):
        """Convert a value from grid units to extent units."""
        size = max(rows, columns) - 1.
        units = extent / size
        return value * units

    def extent_units(self, value):
        """Convert a value from grid units to extent units."""
        raise NotImplementedError

    def create(self):
        """Create the layer in NEST."""
        raise NotImplementedError

    @property
    @if_created
    def gid(self):
        """The NEST global ID (GID) of the layer."""
        return self._gid

    @if_created
    def _connect(self, target, nest_params):
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


class Layer(AbstractLayer):

    def __init__(self, name, params):
        super().__init__(name, params)
        # TODO: use same names
        self.nest_params = {
            'rows': self.params['nrows'],
            'columns': self.params['ncols'],
            'extent': [self.params['visSize']] * 2,
            'edge_wrap': self.params['edge_wrap'],
            'elements': self.build_elements(),
        }
        # TODO: implement
        self._connected = list()
        self._gid = None

    def extent_units(self, value):
        return self.to_extent_units(value, self.visSize, self.nrows, self.ncols)

    def build_elements(self):
        """Return the NEST description of layer elements.

        A NEST element specification is a list of the form::

            [<model_name>, <model_number>, <model_name>, <model_number>, ...]

        This converts the parameters to such a list.
        """
        populations = self.params['populations']
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
        for i, j in itertools.product(range(self.nrows), range(self.ncols)):
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
    def population(self, *args):
        return tuple(self._populations[gid] for gid in args)

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
        print(type(self.stimulator_type))
        # Add a parrot population entry
        populations[self.PARROT_MODEL] = number
        # Make a duplicate sublayer for each filter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        names = filter_suffixes.get_expanded_names(self.name,
                                                   self.params.get('filters'))
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
        for layer in tqdm(self.layers):
            # TODO: Input layers should be able to see different filters
            layer_index = 0
            layer_rates = (float(self.params['max_input_rate'])
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


class ConnectionModel(NestObject):
    """Represent a NEST connection model."""
    pass


class Connection(NestObject):
    """Represent a NEST connection."""

    DEFAULT_SCALE_FACTOR = 1.0

    def __init__(self, source, target, model, params):
        super().__init__(model.name, params)
        self.model = model
        self.source = source
        self.source_population = params.get('source_population', None)
        self.target = target
        self.target_population = params.get('target_population', None)
        # Get NEST connection parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO: Get a view of the kernel, mask, and weights inherited from the
        # connection model

        # Merge 'connection_model' and connection nest_parameters
        nest_params = ChainMap(self.params.get('nest_params', dict()),
                               self.model.params)

        # Get scaling factor, taking in accound whether the connection is
        # convergent or divergent
        if (nest_params['connection_type'] == 'convergent'
                and self.source.params.get('scale_kernels_masks', True)):
            # For convergent connections, the pooling layer is the source
            self.scale_factor = self.source.extent_units(
                self.source.params.get('rf_scale_factor', 1.0)
            )
        elif (nest_params['connection_type'] == 'divergent'
                and self.target.params.get('scale_kernels_masks', True)):
            # For convergent connections, the pooling layer is the target
            self.scale_factor = self.target.extent_units(
                self.target.params.get('rf_scale_factor', 1.0)
            )
        else:
            self.scale_factor = self.DEFAULT_SCALE_FACTOR

        # Set kernel, mask, and weights, scaling if necessary
        nest_params = nest_params.new_child({
            'kernel': self.scale_kernel(nest_params['kernel']),
            'mask': self.scale_mask(nest_params['mask']),
            'weights': self.scale_weights(nest_params['weights']),
        })
        # Set source populations if available
        if self.source_population:
            nest_params['sources'] = {'model': self.source_population}
        if self.target_population:
            nest_params['targets'] = {'model': self.target_population}
        # Save nest_params as a dictionary.
        self.nest_params = dict(nest_params)

    def scale_kernel(self, kernel):
        """Return a new kernel scaled by ``scale_factor``."""
        kernel = deepcopy(kernel)
        try:
            return float(kernel)
        except TypeError:
            if 'gaussian' in kernel:
                kernel['gaussian']['sigma'] *= self.scale_factor
            return kernel

    def scale_mask(self, mask):
        """Return a new mask scaled by ``scale_factor``."""
        mask = deepcopy(mask)
        if 'circular' in mask:
            mask['circular']['radius'] *= self.scale_factor
        if 'rectangular' in mask:
            mask['rectangular'] = {
                key: np.array(scalars) * self.scale_factor
                for key, scalars in mask['rectangular'].items()
            }
        return mask

    def scale_weights(self, weights):
        # Default to no scaling
        gain = self.source.params.get('weight_gain', 1.0)
        return weights * gain

    @if_not_created
    def create(self):
        self.source._connect(self.target, self.nest_params)

    def save(self, output_dir):
        # TODO
        for field in self.params.get('save', []):
            print('TODO: save connection ', field, ' in ', output_dir)

    @property
    def sort_key(self):
        # Mapping for sorting
        return (self.name,
                self.source.name, str(self.source_population),
                self.target.name, str(self.target_population))

    def __lt__(self, other):
        return self.sort_key < other.sort_key


class Population(NestObject):
    """Represents a population.

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
        ntimesteps = int(nest.GetKernelStatus('time')
                          / nest.GetKernelStatus('resolution'))
        formatted_shape = (ntimesteps,) + self.layer.shape
        for unit_index in range(self.number):
            for recorder in self.recorders:
                for variable in recorder.variables:
                    activity = recorder.formatted_data(formatted_shape=formatted_shape,
                                                       variable=variable,
                                                       unit_index=unit_index)
                    filename = save.recorder_filename(self.layer.name,
                                                      self.name,
                                                      unit_index=unit_index,
                                                      variable=variable)
                    save.save_array(join(output_dir, filename), activity)

    def save_rasters(self, output_dir):
        for recorder in self.recorders:
            raster = recorder.get_nest_raster()
            if raster is not None:
                pylab.title(self.layer.name + '_' + self.name)
                f = raster[0].figure
                f.set_size_inches(15, 9)
                filename = ('spikes_raster_' + self.layer.name + '_'
                            + self.name + '.png')
                f.savefig(join(output_dir, filename), dpi=100)
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

    def formatted_data(self, formatted_shape=None, variable=None, unit_index=0):
        return format_recorders.format_recorder(self.gid,
                                                recorder_type=self.type,
                                                shape=formatted_shape,
                                                locations=self.locations,
                                                variable=variable,
                                                unit_index=unit_index)

    def get_nest_raster(self):
        import nest
        from nest import raster_plot
        if (self.type == 'spike_detector'
            and 'memory' in self._record_to
            and len(nest.GetStatus(self.gid)[0]['events']['senders'])):
            return raster_plot.from_device(self.gid, hist=True)
        return None


LAYER_TYPES = {
    None: Layer,
    'InputLayer': InputLayer,
}


class Network:

    def __init__(self, params):
        self._created = False
        self._changed = False
        self.params = params
        # Build network components
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.neuron_models = self.build_named_leaves_dict(
            Model, self.params.c['neuron_models'])
        self.synapse_models = self.build_named_leaves_dict(
            SynapseModel, self.params.c['synapse_models'])
        self.recorder_models = self.build_named_leaves_dict(
            Model, self.params.c['recorders'])
        # Layers can have different types
        self.layers = {
            name: LAYER_TYPES[leaf['type']](name, leaf)
            for name, leaf in self.params.c['layers'].named_leaves()
        }
        self.connection_models = self.build_named_leaves_dict(
            ConnectionModel, self.params.c['connection_models'])
        # Connections must be built last
        self.connections = sorted(
            [
            self.build_connection(connection)
            for connection in self.params.c['topology']['connections']
            ]
        )
        # Populations are represented as a list
        self.populations = sorted(
            self.build_named_leaves_list(
                self.build_population,
                self.params.c['populations']
                )
            )


    @staticmethod
    def build_named_leaves_dict(constructor, node):
        return {name: constructor(name, leaf)
                for name, leaf in node.named_leaves()}

    @staticmethod
    def build_named_leaves_list(constructor, node):
        return [constructor(name, leaf)
                for name, leaf in node.named_leaves()]

    def build_connection(self, params):
        source = self.layers[params['source_layer']]
        target = self.layers[params['target_layer']]
        model = self.connection_models[params['connection']]
        return Connection(source, target, model, params)

    def build_population(self, pop_name, pop_params):
        # Get the gids and locations for the population from the layer object.
        layer = self.layers[pop_params['layer']]
        return Population(pop_name, layer, pop_params)

    def __repr__(self):
        return '{classname}({params})'.format(
            classname=type(self).__name__, params=(self.params))

    def __str__(self):
        return repr(self)

    def _create_all(self, objects):
        for obj in tqdm(objects):
            obj.create()

    def _layer_call(self, method_name, *args, layer_type=None, **kwargs):
        """Call a method on each input layer."""
        for layer in self._get_layers(layer_type=layer_type):
            method = getattr(layer, method_name)
            method(*args, **kwargs)

    def _layer_get(self, attr_name, layer_type=None):
        """Get an attribute from each layer."""
        return tuple(getattr(layer, attr_name)
                     for layer in self._get_layers(layer_type=layer_type))

    def _get_layers(self, layer_type=None):
        if layer_type is None:
            return sorted(self.layers.values())
        return [l for l in sorted(self.layers.values())
                if type(l).__name__ == layer_type]

    @property
    def input_shapes(self):
        return set(self._layer_get('shape', layer_type='InputLayer'))

    @property
    def max_input_shape(self):
        """Max of each dimension."""
        return (max([s[0] for s in self.input_shapes]),
                max([s[1] for s in self.input_shapes]))


    @if_not_created
    def create(self):
        # TODO: use progress bar from PyPhi?
        log.info('Creating neuron models...')
        self._create_all(self.neuron_models.values())
        log.info('Creating synapse models...')
        self._create_all(self.synapse_models.values())
        log.info('Creating recorder models...')
        self._create_all(self.recorder_models.values())
        log.info('Creating layers...')
        self._create_all(self.layers.values())
        log.info('Connecting layers...')
        self._create_all(self.connections)
        log.info('Creating recorders...')
        self._create_all(self.populations)

    def change_synapse_states(self, synapse_changes):
        """Change parameters for some connections of a population.

        Args:
            synapse_changes (list): List of dictionaries each of the form::
                    {synapse_model: <synapse_model>,
                     params: {<key>: <value>,
                              ...}}
                where the params contains the parameters to set for all synapses
                of a given model.

        """
        import nest
        for changes in tqdm(sorted(synapse_changes, key=synapse_sorting_map),
                            desc="-> Change synapses's state."):
            nest.SetStatus(
                nest.GetConnections(synapse_model=changes['synapse_model']),
                changes['params']
                )

    def change_unit_states(self, unit_changes):
        """Change parameters for some units of a population.

        Args:
            unit_changes (list): List of dictionaries each of the form::
                    {'layer': <layer_name>,
                     'population': <pop_name>,
                     'proportion': <prop>,
                     'params': {<param_name>: <param_value>,
                                ...}
                    }
                where <layer_name> and <population_name> define the considered
                population, <prop> is the proportion of units of that population
                for which the parameters are changed, and the ``'params'`` entry is
                the dictionary of parameter changes apply to the selected units.

        """
        import nest
        for changes in tqdm(sorted(unit_changes, key=unit_sorting_map),
                            desc="-> Change units' state"):

            if self._changed and changes['proportion'] == 1:
                raise Exception("Attempting to change probabilistically some" +
                                "units' state multiple times.")

            layer = self.layers[changes['layer']]
            all_gids = layer.gids(population=changes.get('population', None))
            gids_to_change = [all_gids[i] for i
                              in sorted(
                                  random.sample(range(len(all_gids)),
                                                int(len(all_gids)
                                                    * changes['proportion'])
                                                ))]

            nest.SetStatus(gids_to_change,
                           params=changes['params'])
        self._changed = True

    def reset(self):
        import nest
        nest.ResetNetwork()

    def set_input(self, stimulus, start_time=0.):
        self._layer_call('set_input', stimulus, start_time,
                         layer_type='InputLayer')

    def save(self, output_dir, with_rasters=True):
        # Save synapses
        for conn in self.connections:
            conn.save(output_dir)
        # Save recorders
        for population in self.populations:
            population.save(output_dir, with_rasters=with_rasters)

def unit_sorting_map(unit_change):
    """Map by (layer, population, proportion, params_items for sorting."""
    return (unit_change['layer'],
            unit_change['population'],
            unit_change['proportion'],
            sorted(unit_change['params'].items()))

def synapse_sorting_map(synapse_change):
    """Map by (synapse_model, params_items) for sorting."""
    return (synapse_change['synapse_model'],
            sorted(synapse_change['params'].items()))
