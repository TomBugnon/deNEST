#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/models.py

"""Define Models."""

from .nest_object import NestObject
from .utils import if_not_created


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
