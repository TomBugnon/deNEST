#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nestify/build.py

"""Get dependent parameters from independent network parameters."""

# pylint: disable-all

from collections import ChainMap
import itertools
import functools
import logging
import logging.config
from pprint import pformat

import numpy as np
from tqdm import tqdm

from ..utils import filter_suffixes
from ..utils.structures import flatten

log = logging.getLogger(__name__)
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

DUPLICATE_CREATE_WARNING = 'Attempted to create object more than once: %s'
DUPLICATE_CONNECT_WARNING = ('Attempted to connect layers more than once:'
                             'source: %s, target: %s')


def if_not_created(method):
    """Only call a method if the `_created` attribute isn't set.

    After calling, sets the attribute to True.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._created:
            log.warn(DUPLICATE_CREATE_WARNING, self)
            return
        method(self, *args, **kwargs)
        self._created = True
    return wrapper


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

    def _repr_pretty_(self, p, cycle):
        opener = '{classname}({name}, '.format(
            classname=type(self).__name__, name=self.name)
        closer = ')'
        with p.group(p.indentation, opener, closer):
            p.breakable()
            p.pretty(self.params)

    def __repr__(self):
        return '{classname}({name}, {params})'.format(
            classname=type(self).__name__,
            name=self.name,
            params=pformat(self.params))

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)


def calibrate(params):
    """Calculate dependent parameters from independent network parameters.

    Returns:
        Params: A new set of parameters, with all dependent parameters
        included.
    """
    return {
        'neuron_models': models(params.c['neuron_models']),
        'synapse_models': models(params.c['synapse_models'])
    }


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
            if not 'target_neuron' in params:
                raise ValueError("must specify 'target_neuron' "
                                 "if providing 'receptor_type'")
            import nest
            target = self.nest_params.pop('target_neuron')
            receptors = nest.GetDefaults(target)['receptor_types']
            self.nest_params['receptor_type'] = receptors[self.params['receptor_type']]


class Layer(NestObject):

    def __init__(self, name, params):
        super().__init__(name, params)
        # TODO: use same names
        self.nest_params = { 'rows': self.params['nrows'],
            'columns': self.params['ncols'],
            'extent': [self.params['visSize']] * 2,
            'edge_wrap': self.params['edge_wrap'],
            'elements': self.build_elements(),
        }
        self._connected = list()

    def build_elements(self):
        """Return the NEST description of layer elements.

        A NEST element specification is a list of the form::

            [<model_name>, <model_number>, <model_name>, <model_number>, ...]

        This converts the parameters to such a list.
        """
        element_params = self.params['elements']
        default_ratios = self.params['ratios']
        # Map types to numbers
        types_to_numbers = {element['type']: element.get('number')
                            for element in element_params.values()}

        def calculate_number(params):
            if 'number' in params:
                return params['number']
            # Use default ratio if there's no ratio key
            relative_to, ratio = params.get('ratio',
                                            default_ratios[params['type']])
            # Calculate the number based on the ratio relative to the given
            # type
            return ratio * types_to_numbers[relative_to]

        return flatten([population, calculate_number(params)]
                       for population, params in element_params.items())

    @if_not_created
    def create(self):
        from nest import topology as tp
        self.gid = tp.CreateLayer(self.nest_params)

    def _connect(self, target, nest_params):
        # NOTE: Don't use this method directly; use a Connection instead
        from nest import topology as tp
        tp.ConnectLayers(self.gid, target.gid, nest_params)


class InputLayer(NestObject):
    """A layer that provides input to the network.

    This layer consists of several sublayers, each distinct NEST topological
    layers, represented by a ``Layer``:
        - For each filter combination, a stimulator/parrot pair of layers is
        created.
        - In such a pair, the stimulator layer contains stimulator devices,
          while the parrot layer passes the stimuli from the stimulator layer
          to multiple outputs.
    """

    PARROT_SUFFIX = '_parrot'
    PARROT_MODEL = 'parrot_neuron'

    def __init__(self, name, params):
        super().__init__(name, params)
        # Make a duplicate layer for each filter
        names = filter_suffixes.get_expanded_names(self.name,
                                                   self.params.get('filters'))
        self.stimulators = [Layer(name, self.params) for name in names]
        # Make copies with parrot neurons
        self.parrots = [self.build_parrot(layer) for layer in self.stimulators]

    @property
    def layers(self):
        """Stimulator/parrot layer pairs."""
        return zip(self.stimulators, self.parrots)

    def build_parrot(self, stimulator):
        """Return a layer of parrot neurons mimicking a stimulator layer."""

        def replace_with_parrot(element_params):
            if len(element_params) != 1:
                raise ValueError('input layer must have only one element type')
            value = list(element_params.values())[0]
            return {self.PARROT_MODEL: value}

        # Override layer elements with parrot neurons
        params = ChainMap(
            {'elements': replace_with_parrot(stimulator.params['elements'])},
            stimulator.params
        )
        return Layer(stimulator.name + self.PARROT_SUFFIX, params)

    def create(self):
        # Create and connect sublayers
        for stimulator, parrot in self.layers:
            stimulator.create()
            parrot.create()
            stimulator

    def _connect(self, target, nest_params):
        # NOTE: Don't use this method directly; use a Connection object instead
        from nest import topology as tp
        tp.ConnectLayers(self.gid, target.gid, nest_params)


class ConnectionModel(NestObject):
    """Represent a NEST connection model."""
    pass


class Connection(NestObject):
    """Represent a NEST connection."""

    DEFAULT_RF_FACTOR = 1.0

    def __init__(self, source, target, model, params):
        super().__init__(model.name, params)
        self.model = model
        self.source = source
        self.source_population = params.pop('source_population')
        self.target = target
        self.target_population = params.pop('target_population')
        self.source_size = max(source.params['nrows'], source.params['ncols'])
        # TODO: RF and weight scaling:
        # - maskfactor?
        # - btw error in ht files: secondary horizontal intralaminar mixes dcpS
        #   and dcpP
        if self.target.params.get('scale_kernels_masks'):
            self.rf_factor = (self.target.params.get('rf_scale_factor', 1.0) *
                              self.source.params['visSize'] / self.source_size)
        else:
            self.rf_factor = self.DEFAULT_RF_FACTOR
        # Get NEST connection parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get a view of the kernel, mask, and weights inherited from the
        # connection model
        nest_params = ChainMap(self.params.get('nest_params', dict()),
                               self.model.params)
        # Set sources, targets, and scale if necessary
        nest_params = {
            'sources': {'model': self.source_population},
            'targets': {'model': self.target_population},
            'kernel': self.scale_kernel(nest_params['kernel']),
            'mask': self.scale_mask(nest_params['mask']),
            'weights': self.scale_weights(nest_params['weights']),
        }
        # Inherit other properties from the connection model as well
        self.nest_params = dict(ChainMap(nest_params, self.model.params))

    def scale_kernel(self, kernel):
        try:
            return float(kernel)
        except TypeError:
            if 'gaussian' in kernel:
                kernel['gaussian']['sigma'] *= self.rf_factor
            return kernel

    def scale_mask(self, mask):
        if 'circular' in mask:
            mask['circular']['radius'] *= self.rf_factor
        if 'rectangular' in mask:
            mask['rectangular'] = {
                key: np.array(scalars) * self.rf_factor
                for key, scalars in mask['rectangular'].items()
            }
        return mask

    def scale_weights(self, weights):
        # Default to no scaling
        gain = self.source.params.get('weight_gain', 1)
        return weights * gain

    @if_not_created
    def create(self):
        from nest import topology as tp
        if isinstance(self.source, InputLayer):
            return
        self.source._connect(self.target, self.nest_params)


LAYER_TYPES = {
    None: Layer,
    'InputLayer': InputLayer,
}


def build_named_leaves(constructor, node):
    return {name: constructor(name, leaf) for name, leaf in node.named_leaves()}


class Network:

    def __init__(self, params):
        self._created = False
        self.params = params
        # Build network components
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.neuron_models = build_named_leaves(
            Model, self.params.c['neuron_models'])
        self.synapse_models = build_named_leaves(
            SynapseModel, self.params.c['synapse_models'])
        # Layers can have different types
        self.layers = {
            name: LAYER_TYPES[leaf['type']](name, leaf)
            for name, leaf in self.params.c['layers'].named_leaves()
        }
        self.connection_models = build_named_leaves(
            ConnectionModel, self.params.c['connection_models'])
        # Connections must be built last
        self.connections = [
            self.build_connection(connection)
            for connection in self.params.c['topology']['connections']
        ]

    def build_connection(self, params):
        source = self.layers[params.pop('source_layer')]
        target = self.layers[params.pop('target_layer')]
        model = self.connection_models[params.pop('connection')]
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
        # TODO: use progress bar from pyphi
        log.info('Creating neuron models...')
        self._create_all(self.neuron_models.values())
        log.info('Creating synapse models...')
        self._create_all(self.synapse_models.values())
        log.info('Creating layers...')
        self._create_all(self.layers.values())
        log.info('Connecting layers...')
        self._create_all(self.connections)
