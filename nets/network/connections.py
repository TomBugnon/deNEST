#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py
"""ConnectionModel and Connection objects."""

# pylint:disable=missing-docstring

from ..base_object import NestObject
from .utils import if_not_created


class ConnectionModel(NestObject):
    """Represent a NEST connection model.

    Connection objects inherit ``nest_params`` and ``params`` attributes from
    ConnectionModel objects.

    Args:
        name (str): Name of the connection model.
        params (dict-like): Dictionary of parameters. The following parameters
            are recognized:
                type (str): Type of connection. Currently, only connections of
                    type 'topological' are supported
        nest_params (dict-like): Dictionary of parameters that will be passed
            to NEST during the ``tp.ConnectLayer`` call. The following
            parameters are mandatory: ``['synapse_model'``. The ``sources`` and
            ``targets`` NEST parameters are reserved. The source and target
            populations are set by ``Connection`` objects via the
            ``source_population`` and ``target_population`` connection
            parameters.
    """

    # Validation of `params`
    RESERVED_PARAMS = []
    MANDATORY_PARAMS = []
    OPTIONAL_PARAMS = {
        'type': 'topological'
    }
    # Validation of `nest_params`
    RESERVED_NEST_PARAMS = ['sources', 'targets']
    MANDATORY_NEST_PARAMS = ['synapse_model']
    OPTIONAL_NEST_PARAMS = None

    def __init__(self, name, params, nest_params):
        super().__init__(name, params, nest_params)
        self._type = self.params['type']
        # Check that the connection types are recognized and nothing is missing.
        assert self.type in ['topological']

    @property
    def type(self):
        return self._type


class BaseConnection(NestObject):
    """Base class for all population-to-population connections.

    A Connection consists in synapses between two populations that have a
    specific connection model. Population-to-population connections are
    specified in the ``connections`` network/topology parameter. Connection
    models are specified in the ``connection_models`` network parameter.

    ``(<connection_model_name>, <source_layer_name>, <source_population_name>,
    <target_layer_name>, <target_population_name>)`` tuples fully specify each
    individual connection and should be unique. Refer to
    ``Network.build_connections`` for a description of how these arguments are
    parsed.

    Connection weights can be recorded by 'weight_recorder' devices,
    represented by ConnectionRecorder objects. Because the weight recorder
    device's GID must be specified in a synapse model's default parameters
    (using nest.SetDefaults or nest.CopyModel), the actual NEST synapse model
    used by a connection (`self.nest_synapse_model`) might be different from the
    one specified in the connection's parameters (`self.base_synapse_model`)

    Args:
        model (``ConnectionModel``): ``ConnectionModel`` object. The ``params``
            and ``nest_params`` parameter dictionaries are inherited from the
            ``ConnectionModel`` object.
        source_layer, target_layer (``Layer``): source and target Layer object
        source_population, target_population (str | None): Name of the
            source and target population. If None, all populations are used.
            Wrapper for the ``sources`` and ``targets`` ``nest.ConnectLayers``
            parameters.
    """

    # TODO: Make some methods private?
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(self, model, source_layer, source_population, target_layer,
                 target_population):
        """Initialize Connection object."""
        # Inherit params and nest_params from connection model
        super().__init__(model.name, model.params, model.nest_params)
        ##
        # Define the source and targets
        self.model = model
        self.source = source_layer
        self.target = target_layer
        self.source_population = source_population
        if self.source_population:
            self.nest_params['sources'] = {'model': self.source_population}
        self.target_population = target_population
        if self.target_population:
            self.nest_params['targets'] = {'model': self.target_population}
        #
        # Base synapse model is retrieved from nest_params. If a weight_recorder
        # is created for this connection, a different synapse model will be used
        self._base_synapse_model = self.nest_params['synapse_model']
        self._nest_synapse_model = self.nest_params['synapse_model']
        self.validate_connection()

    # Properties:
    @property
    def base_synapse_model(self):
        """Return synapse model specified in Connection's model."""
        return self._nest_synapse_model

    @property
    def nest_synapse_model(self):
        """Return synapse model used in NEST for this connection.

        May differ from self.base_synapse_model to allow recording to a
        weight_recorder.
        """
        return self._nest_synapse_model

    def __str__(self):
        return '-'.join([
            self.name,
            self.source.name, str(self.source_population),
            self.target.name, str(self.target_population),
        ])

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    # Creation and connection

    @if_not_created
    def create(self):
        """Create the connections in NEST and the connection recorders."""
        pass

    def connect_connection_recorder(self, recorder_type='weight_recorder',
                                    recorder_gid=None):
        """Create and use new synapse model connected to a ConnectionRecorder.

        The new `nest_synapse_model` will be used during creation rather than
        the `base_synapse_model` specified in the nest parameters.
        """
        import nest
        if recorder_type == 'weight_recorder':
            # If we want to record the synapse, we create a new model and change
            # its default params so that it connects to the weight recorder
            self._nest_synapse_model = self.nest_synapse_model_name()
            nest.CopyModel(
                self._base_synapse_model,
                self._nest_synapse_model,
                {
                    recorder_type: recorder_gid
                }
            )
            # Use modified synapse model for connection
            self.nest_params['synapse_model'] = self._nest_synapse_model
        else:
            raise ValueError(f"ConnectionRecorder type `{recorder_type}` not"
                             "recognized")

    def nest_synapse_model_name(self):
        return f"{self._base_synapse_model}-{self.__str__()}"

    # Save and plot stuff
    def save(self, output_dir):
        pass

    def validate_connection(self):
        """Check the connection to avoid bad errors.

        Raise ValueError if:
            1. ``InputLayer`` layers are never targets
            2. The source population of ``InputLayer`` layers is
                ``parrot_neuron``
        """

        if type(self.target).__name__ == 'InputLayer':
            raise ValueError(
                f"Invalid target in connection {str(self)}: `InputLayer` layers"
                f" cannot be connection targets."
            )
        if (
            type(self.source).__name__ == 'InputLayer'
            and self.source_population != self.source.PARROT_MODEL
        ):
            raise ValueError(
                f"Invalid source population for connection {str(self)}: "
                f" the source population of connections must be"
                f"{self.source.PARROT_MODEL} for `InputLayer` layers"
            )


class TopoConnection(BaseConnection):
    """Represent a topological connection."""

    def __init__(self, *args):
        super().__init__(*args)

    # Creation functions not inherited from BaseConnection

    @if_not_created
    def create(self):
        """Create the connections in NEST using ``tp.ConnectLayers``."""
        self.source.connect(self.target, self.nest_params)
