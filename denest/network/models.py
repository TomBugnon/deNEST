#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/models.py

"""NEST model classes."""

from ..base_object import NestObject
from ..utils.validation import MissingParameterError, ReservedParameterError
from .utils import if_not_created


class Model(NestObject):
    """Represent a model in NEST.

    During creation, we copy or change the default parameters of the
    ``nest_model`` NEST model.

    Args:
        name (str): Name of the model
        params (dict-like): `params` of the object. Should countain the
            `nest_model` key.
        nest_params (dict-like): Dictionary passed to NEST during the
            ``nest.CopyModel`` of ``nest.SetDefaults`` call.
    """

    # Validation of `params`
    RESERVED_PARAMS = []
    MANDATORY_PARAMS = ["nest_model"]
    OPTIONAL_PARAMS = {}
    # Validation of `nest_params`
    RESERVED_NEST_PARAMS = None
    MANDATORY_NEST_PARAMS = None
    OPTIONAL_NEST_PARAMS = None

    def __init__(self, name, params, nest_params):
        super().__init__(name, params, nest_params)
        # Save and remove the NEST model name from the nest parameters.
        self.nest_model = self.params["nest_model"]

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

    Args:
        name (str): Name of the model
        params (dict-like): `params` of the object. Should countain the
            `nest_model` key. The following keys are recognized:
                - ``receptor_type``, ``target_neuron`` (str): Name of the
                    receptor type (eg "AMPA") and of the target neuron for
                    synapses of this type. If specified, the `receptor_type`
                    NEST parameter (which is an integer) is automatically set
                    from the defaults of the target neuron.
        nest_params (dict-like): Dictionary passed to NEST during the
            ``nest.CopyModel`` of ``nest.SetDefaults`` call. The ``weight``
            parameter, which sets the synapse model's default weight is
            reserved. To set the strength of projections, set the ``weights``
            parameter of ``Projection`` objects instead.

    ..note::
        NEST expects 'receptor_type' to be an integer rather than a string. The
        integer index must be found in the defaults of the target neuron.
    """

    # Validation of `nest_params`
    RESERVED_NEST_PARAMS = ["weight"]

    def __init__(self, name, params, nest_params):
        # Get receptor index in NEST from receptor name
        if ("receptor_type" in params and "target_neuron" not in params) or (
            "target_neuron" in params and "receptor_type" not in params
        ):
            raise MissingParameterError(
                f"Missing parameter in SynapseModel {name} `params`: "
                f"Must specify both `target_neuron` and `receptor_type` params"
                f" or neither of them"
            )
        if "receptor_type" in params:
            import nest

            target = params.pop("target_neuron")
            receptor_name = params.pop("receptor_type")
            receptor_ids = nest.GetDefaults(target)["receptor_types"]
            if "receptor_type" in nest_params:
                raise ReservedParameterError(
                    f"Reserved parameter in SynapseModel {name} `nest_params`: "
                    f"`receptor_type` should not be specified in nest_params if"
                    f"specified in params."
                )
            nest_params["receptor_type"] = receptor_ids[receptor_name]
        # Initialize Model
        super().__init__(name, params, nest_params)
