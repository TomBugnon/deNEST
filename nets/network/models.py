#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/models.py

"""NEST model classes."""

from .nest_object import NestObject
from .utils import if_not_created

# Substrings used to guess the type (exc/inh) of synapses
EXC_SUBSTRINGS = ['AMPA', 'NMDA', 'exc']
INH_SUBSTRINGS = ['GABA', 'inh']

class Model(NestObject):
    """Represent a model in NEST."""

    # pylint:disable=too-few-public-methods

    def __init__(self, name, params):
        super().__init__(name, params)
        # Save and remove the NEST model name from the nest parameters.
        self.nest_model = self.params.pop('nest_model')
        # Get the model's "type" (+1 for excitatory neuron/synapse,
        # -1 for inhibitory)
        self.type = self.params.pop('type', None)
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
        # Try to infer the synapse type (1 for excitatory/-1 for inhibitory)
        # If not specified in the params.
        if self.type is None:
            self.type = self.guess_synapse_type()

    def guess_synapse_type(self):
        """Guess the synapse type from the model or receptor name.

        Return +1(/-1) for excitatory (/inhibitory) synapse.
        """

        syn_type = None
        # Try to match either the receptor type or the model name
        if 'receptor_type' in self.params:
            test_string = self.params['receptor_type']
        else:
            test_string = self.name
        if any(substring.lower() in test_string.lower()
               for substring in EXC_SUBSTRINGS):
            syn_type = 1.0
        elif any(substring.lower() in test_string.lower()
               for substring in INH_SUBSTRINGS):
            syn_type = -1.0

        # Tell USER about it
        if syn_type is None:
            print(f'Could not guess synapse type for synapse {self.name}. '
                  f'You can specify the synapse type (+-1 for exc/inh) in the'
                  f' ``type`` field of the parameters.')
        else:
            print(f'Guessing the type: `{syn_type}`\t'
                  f'...for synapse:\t {self.name}.')

        return syn_type
