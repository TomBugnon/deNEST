#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py

"""Define Connections objects"""

from collections import ChainMap
from copy import deepcopy

import numpy as np

from .utils import NestObject, if_created, if_not_created


class ConnectionModel(NestObject):
    """Represent a NEST connection model."""
    DEFAULT_SCALE_FACTOR = 1.0

    def __init__(self, name, params):
        super().__init__(name, params)
        self._scale_factor = self.params.pop('scale_factor',
                                             self.DEFAULT_SCALE_FACTOR)

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

    @property
    def sort_key(self):
        # Mapping for sorting
        return (self.name,
                self.source.name, str(self.source_population),
                self.target.name, str(self.target_population))

    def __lt__(self, other):
        return self.sort_key < other.sort_key
