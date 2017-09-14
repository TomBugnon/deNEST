#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/build.py

"""Get dependent parameters from independent network parameters."""

# pylint: disable=too-few-public-methods

from collections import ChainMap
import itertools
import functools
import logging
import logging.config
from pprint import pformat

import numpy as np
from tqdm import tqdm

from ..utils import filter_suffixes

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
        """Create the NEST object represented by this model."""
        import nest
        nest.CopyModel(self.nest_model, self.name, self.nest_params)


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

    @staticmethod
    def to_grid_units(value, extent, rows, columns):
        size = max(rows, columns)
        units = extent / size
        return value * units

    def grid_units(self, value):
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

    def grid_units(self, value):
        return self.to_grid_units(value, self.visSize, self.nrows, self.ncols)

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
        return tuple(filter(loc_filt, filter(pop_filt, self._gids)))

    @if_created
    def element(self, *args):
        return tuple(self._elements[location] for location in args)

    @if_created
    def location(self, *args):
        return tuple(self._locations[gid] for gid in args)

    @if_created
    def population(self, *args):
        return tuple(self._populations[gid] for gid in args)


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
        # Add a parrot population entry
        populations[self.PARROT_MODEL] = number
        # Make a duplicate sublayer for each filter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        names = filter_suffixes.get_expanded_names(self.name,
                                                   self.params.get('filters'))
        self.layers = [Layer(name, self.params) for name in names]

    def grid_units(self, value):
        # IMPORTANT: Assumes all sublayers are the same size!
        return self.layers[0].grid_units(value)

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
        # Create sublayers
        self._layer_call('create')
        # Set the GID
        self._gid = flatten(self._layer_get('gid'))
        # Connect stimulators to parrots, one-to-one
        # IMPORTANT: This assumes that all layers are the same size!
        radius = self.grid_units(0.1)
        one_to_one_connection = {
            'sources': {'model': self.stimulator_model},
            'targets': {'model': self.PARROT_MODEL},
            'connection_type': 'convergent',
            'synapse_model': 'static_synapse',
            'mask': {'circular': {'radius': radius}}
        }
        tp.ConnectLayers(self._gid, self._gid, one_to_one_connection)

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
        # TODO: RF and weight scaling:
        # - maskfactor?
        # - btw error in ht files: secondary horizontal intralaminar mixes dcpS
        #   and dcpP
        if self.target.params.get('scale_kernels_masks'):
            self.scale_factor = self.source.grid_units(
                self.target.params.get('rf_scale_factor', 1.0)
            )
        else:
            self.scale_factor = self.DEFAULT_SCALE_FACTOR
        # Get NEST connection parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO: handle divergent and convergent separately
        # Get a view of the kernel, mask, and weights inherited from the
        # connection model
        nest_params = ChainMap(self.params.get('nest_params', dict()),
                               self.model.params)
        # Set kernel, mask, and weights, scaling if necessary
        nest_params = {
            'kernel': self.scale_kernel(nest_params['kernel']),
            'mask': self.scale_mask(nest_params['mask']),
            'weights': self.scale_weights(nest_params['weights']),
        }
        # Set source populations if available
        if self.source_population:
            nest_params['sources'] = {'model': self.source_population}
        if self.target_population:
            nest_params['targets'] = {'model': self.target_population}
        # Inherit other properties from the connection model as well
        self.nest_params = dict(ChainMap(nest_params, self.model.params))

    def scale_kernel(self, kernel):
        try:
            return float(kernel)
        except TypeError:
            if 'gaussian' in kernel:
                kernel['gaussian']['sigma'] *= self.scale_factor
            return kernel

    def scale_mask(self, mask):
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


LAYER_TYPES = {
    None: Layer,
    'InputLayer': InputLayer,
}


class Network:

    def __init__(self, params):
        self._created = False
        self.params = params
        # Build network components
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.neuron_models = self.build_named_leaves(
            Model, self.params.c['neuron_models'])
        self.synapse_models = self.build_named_leaves(
            SynapseModel, self.params.c['synapse_models'])
        # Layers can have different types
        self.layers = {
            name: LAYER_TYPES[leaf['type']](name, leaf)
            for name, leaf in self.params.c['layers'].named_leaves()
        }
        self.connection_models = self.build_named_leaves(
            ConnectionModel, self.params.c['connection_models'])
        # Connections must be built last
        self.connections = [
            self.build_connection(connection)
            for connection in self.params.c['topology']['connections']
        ]

    @staticmethod
    def build_named_leaves(constructor, node):
        return {name: constructor(name, leaf)
                for name, leaf in node.named_leaves()}

    def build_connection(self, params):
        source = self.layers[params['source_layer']]
        target = self.layers[params['target_layer']]
        model = self.connection_models[params['connection']]
        return Connection(source, target, model, params)

    def __repr__(self):
        return '{classname}({params})'.format(
            classname=type(self).__name__, params=(self.params))

    def __str__(self):
        return repr(self)

    def _create_all(self, objects):
        for obj in tqdm(objects):
            obj.create()

    @if_not_created
    def create(self):
        # TODO: use progress bar from PyPhi?
        log.info('Creating neuron models...')
        self._create_all(self.neuron_models.values())
        log.info('Creating synapse models...')
        self._create_all(self.synapse_models.values())
        log.info('Creating layers...')
        self._create_all(self.layers.values())
        log.info('Connecting layers...')
        self._create_all(self.connections)
