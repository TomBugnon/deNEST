#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py

"""Connection classes."""

import csv
from collections import ChainMap
from copy import deepcopy
from os.path import join


from tqdm import tqdm

from .nest_object import NestObject
from .utils import if_not_created


class ConnectionModel(NestObject):
    """Represent a NEST connection model."""
    DEFAULT_SCALE_FACTOR = 1.0

    def __init__(self, name, params):
        super().__init__(name, params)
        self._scale_factor = self.params.pop('scale_factor',
                                             self.DEFAULT_SCALE_FACTOR)
        self._dump_connection = self.params.pop('dump_connection', False)

    @property
    def dump_connection(self):
        return self._dump_connection

    @property
    def scale_factor(self):
        return self._scale_factor


class Connection(NestObject):
    """Represent a NEST connection."""

    def __init__(self, source, target, model, params):
        super().__init__(model.name, params)
        self.model = model
        self.source = source
        self.source_population = params.get('source_population', None)
        self.target = target
        self.target_population = params.get('target_population', None)
        self.scale_factor = None
        self.nest_params = self.get_nest_params()

    def get_nest_params(self):
        # Get NEST connection parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # TODO: Get a view of the kernel, mask, and weights inherited from the
        # connection model

        # Merge 'connection_model' and connection nest_parameters
        # IMPORTANT: The connection model parameters are shared between
        # different connections, so we need to make our own copy before
        # modifying anything in-place.
        nest_params = deepcopy(
            ChainMap(self.params.get('nest_params', dict()),
                     self.model.params)
        )

        self.scale_factor = self.model.scale_factor
        # Get connection-specific scaling factor, taking in account whether the
        # connection is convergent or divergent
        if (nest_params['connection_type'] == 'convergent'
                and self.source.params.get('scale_kernels_masks', True)):
            # For convergent connections, the pooling layer is the source
            self.scale_factor = self.source.extent_units(self.scale_factor)
        elif (nest_params['connection_type'] == 'divergent'
                and self.target.params.get('scale_kernels_masks', True)):
            # For convergent connections, the pooling layer is the target
            self.scale_factor = self.target.extent_units(self.scale_factor)

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
        # Return nest_params as a dictionary.
        return dict(nest_params)

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


    def dump(self, dump_dir):
        # TODO: Query using synapse labels to identify connections with same
        # source pop, target pop and synapse model
        if self.model.dump_connection:
            conns = nest.GetConnections(
                source=self.source.gids(population=self.source_population),
                target=self.target.gids(population=self.target_population),
                synapse_model=self.nest_params['synapse_model'])
            # We save: source_gid, target_gid, synapse_model, weight, delay
            with open(join(dump_dir, self.__str__), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(format_dump(conns))

    @property
    def __str__(self):
        return '-'.join(self.sort_key)

    @property
    def sort_key(self):
        # Mapping for sorting
        return (self.name,
                self.source.name, str(self.source_population),
                self.target.name, str(self.target_population))

    def __lt__(self, other):
        return self.sort_key < other.sort_key


def format_dump(conns):
    import nest
    formatted = []
    for conn in conns:
        status = nest.GetStatus((conn,))[0]
        formatted.append((status['source'],
                          status['target'],
                          str(status['synapse_model']),
                          status['weight'],
                          status['delay']))
    return formatted
