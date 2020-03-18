#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# recorders.py

"""Create and save population and connection recorder objects."""

# pylint:disable=missing-docstring

from copy import deepcopy

from .. import save
from .nest_object import NestObject
from .utils import if_created, if_not_created

POP_RECORDER_TYPES = ['multimeter', 'spike_detector']

# Parameters that are recognized and not passed as NEST parameters. These
# parameters are set in `populations.yml` rather than `recorders.yml`
NON_NEST_PARAMS = {}


class BaseRecorder(NestObject):
    """Base class for all recorder classes. Represent nodes (not models).

    NB: The recorder models are created separately !!! All the parameters
    passed to these classes originate not from the `recorder` parameters
    (`recorders.yml`) but from the population parameters !
    IMPORTANT: The parameters from the recorder models ('recorders.yml') are set
    as defaults of the NEST model, and the parameters of the instances of this
    class overwrite those defaults at creation.

    TODO: Don't create the recorder models separately so we can define
    "non_nest" parameters in `recorder.yml`.
    """
    def __init__(self, name, all_params):
        # Pop off the params that shouldn't be considered as NEST parameters
        nest_params = deepcopy(dict(all_params))
        params = {}
        for non_nest_param, default in NON_NEST_PARAMS.items():
            params[non_nest_param] = nest_params.pop(non_nest_param, default)
        # We now save the params and nest_params dictionaries as attributes
        super().__init__(name, params)
        self.nest_params = nest_params
        self._type = None
        # Attributes below may depend on NEST default and recorder models and
        # are updated after creation
        self._gid = None  # gid of recorder node
        self._filenames = None
        self._record_to = None  # eg ['memory', 'file']
        self._withtime = None
        self._label = None  # Only affects raw data filenames.

    @property
    @if_created
    def gid(self):
        return self._gid

    @property
    @if_created
    def variables(self):
        return self._record_from

    @property
    def type(self):
        return self._type

    def __str__(self):
        raise NotImplementedError

    def set_status(self, params):
        """Call nest.SetStatus to set recorder params."""
        import nest
        print(f'--> Setting status for recorder {self.name}: {params}')
        nest.SetStatus(self.gid, params)

    def set_label(self):
        """Set self._label attribute and set nest parameter accordingly."""
        import nest
        self._label = self.__str__()
        # Don't use self.set_status to avoid verbose
        # self.set_status({'label': self._label})
        nest.SetStatus(self.gid, {'label': self._label})

    def raw_data_colnames(self):
        """Return list of labels for columns in raw data saved by NEST."""
        raise NotImplementedError

    def raw_data_filenames(self):
        """Return names of raw data files saved by NEST.

        From NEST documentation:
            ```The name of the output file is
            `data_path/data_prefix(label|model_name)-gid-vp.file_extension`

            See /label and /file_extension for how to change the name.
            /data_prefix is changed in the root node.```

        NB: We don't use the recorder's `filenames` key, since it is created
        only after the first `Simulate` call.

        NB: The label is set at creation.

        NB: There is one file per virtual process. The virtual processes are
        numeroted from 0 and formatted with the same number of digits as that of
        the number of virtual processes.
        """
        import nest
        # TODO: Deal with case where multimeter is only recorded to memory?
        assert 'file' in self._record_to
        assert self._label is not None  # Check that the label has been set
        prefix = (nest.GetKernelStatus('data_prefix')
                  + self._label
                  + f'-{self.gid[0]}-')
        extension = nest.GetStatus(self.gid, 'file_extension')[0]
        n_vp = nest.GetKernelStatus('local_num_threads')
        # TODO: CHeck the formatting for 3 digits number of threads
        assert n_vp <= 100
        n_digits = len(str(n_vp))
        return [
            prefix + f'{str(vp).zfill(n_digits)}.{extension}'
            for vp in range(n_vp)
        ]

    @if_created
    def get_base_metadata_dict(self):
        """Return dictionary containing metadata for all recorder types."""
        return {
            'type': self._type,
            'label': self._label,
            'colnames': self.raw_data_colnames(),
            'filenames': self.raw_data_filenames(),
        }

    def save_metadata(self):
        """Save metadata for recorder."""
        raise NotImplementedError


class PopulationRecorder(BaseRecorder):
    """Represent a recorder node. Connects to a single population.

    Handles connecting the recorder node to the population.
    The recorder objects contains the population and layer specs necessary for
    creating recorder output metadata (population shape, unit locations, etc).

    Args:
        model (str): Model of recorder (eg 'multimeter')
        layer (Layer): Layer object
        population_name (str): Name of population to connect.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, model, layer, population_name):
        """Initialize PopulationRecorder object."""
        super().__init__(model, {})
        self.layer = layer
        self._population_name = population_name  # Name of recorded population
        self._layer_name = self.layer.name  # Name of recorded pop's layer
        self._layer_shape = self.layer.shape  # (nrows, cols) for recorded pop
        # TODO
        self._units_number = None  # Number of nodes per grid position for pop
        # Attributes below are necessary for creating the output metadata file
        # and are defined during the `self.create()` call.
        self._gids = None  # all gids of recorded nodes
        self._locations = None  # location by gid for recorded nodes
        ##
        self._type = None  # 'spike detector' or 'multimeter'
        if self.name in POP_RECORDER_TYPES:
            self._type = self.name
        else:
            # We try to guess the recorder type from its name
            self._type = self.guess_rec_type()
            print(f'Guessing type for recorder `{self.name}`:'
                  f' `{self._type}`')
        # Attributes below may depend on NEST default and recorder models and
        # are updated after creation
        self._record_from = None  # list of variables for mm, or ['spikes']
        self._interval = None  # Sampling interval. Ignored for spike_detector.

    @property
    def population_name(self):
        """Return name of population the recorder is connected to."""
        return self._population_name

    @property
    def layer_name(self):
        """Return name of layer the recorder is connected to."""
        return self._layer_name

    def guess_rec_type(self):
        """Guess recorder type from recorder name."""
        # TODO: add to doc
        # TODO: access somehow the base nest model from which the recorder
        # model inherits rather than guessing
        for rec_type in POP_RECORDER_TYPES:
            if rec_type in self.name:
                return rec_type
        raise Exception("The type of recorder {self.name} couldn't be guessed")

    @if_not_created
    def create(self):
        """Create the PopulationRecorder node

        1. Create the recorder node in NEST
        2. Obtain the PopulationRecorder's attributes values from the Layer
            object and the node in NEST
        3. Connect the node to the target GIDs.
        """
        import nest
        # Create node
        self._gid = nest.Create(self.name, params={})
        # Save population and layer-wide attributes
        self._gids = self.layer.gids(population=self.population_name)
        self._locations = None  # TODO
        # Update attributes after creation (may depend on nest defaults and
        # recorder models)
        self._record_to = nest.GetStatus(self.gid, 'record_to')[0]
        self._withtime = nest.GetStatus(self.gid, 'withtime')[0]
        if self.type == 'multimeter':
            self._record_from = [str(variable) for variable
                                 in nest.GetStatus(self.gid, 'record_from')[0]]
            self._interval = nest.GetStatus(self.gid, 'interval')[0]
        elif self.type == 'spike_detector':
            self._record_from = ['spikes']
        # Set the label and filename prefix
        self.set_label()
        self._filenames = self.raw_data_filenames
        # Connect population
        if self.type == 'multimeter':
            nest.Connect(self.gid, self.gids)
        elif self.type == 'spike_detector':
            nest.Connect(self.gids, self.gid)

    @property
    @if_created
    def gids(self):
        return self._gids

    @property
    @if_created
    def locations(self):
        return self._locations

    def __str__(self):
        return self.type + '_' + self._layer_name + '_' + self._population_name

    def get_population_recorder_metadata_dict(self):
        metadata_dict = self.get_base_metadata_dict()
        metadata_dict.update({
            'gids': self._gids,
            'locations': self._locations,
            'population_name': self._population_name,
            'layer_name': self._layer_name,
            'layer_shape': self._layer_shape,
            'population_shape': self._layer_shape + (self._units_number,),
            'units_number': self._units_number,
            'interval': self._interval,
            'record_from': self._record_from,
        })
        return metadata_dict

    def save_metadata(self, output_dir):
        metadata_path = save.output_path(
            output_dir,
            'recorders_metadata',
            self._label
        )
        save.save_as_yaml(metadata_path,
                          self.get_population_recorder_metadata_dict())

    def raw_data_colnames(self):
        import nest
        # TODO: Make sure this is necessary or find a workaround?
        # TODO: check the parameters at creation rather than here
        assert nest.GetStatus(self.gid, 'withtime')[0]
        if self.type == 'multimeter':
            return ['gid', 'time'] + self._record_from
        elif self.type == 'spike_detector':
            return ['gid', 'time']
        else:
            raise Exception


class ConnectionRecorder(BaseRecorder):
    """Represents a recorder connected to synapses of a ``Connection``.

    ConnectionRecorders are connected to synapses of a single
    population-to-population ``Connection`` object.
    
    Args:
        model (str): Model of the connection recorder (eg 'weight_recorder')
        connection (``Connection``): ``Connection`` object the recorder is
            connected to.
    """

    # pylint:disable=too-many-instance-attributes
    CONNECTION_RECORDER_TYPES = ['weight_recorder']

    def __init__(self, model, connection):
        super().__init__(model, {})
        self._model = model
        self._connection = connection
        self._connection_name = str(connection)
        # "type" or ConnectionRecorder ("weight_recorder")
        self._type = None 
        for type in self.CONNECTION_RECORDER_TYPES:
            if type in self._model:
                self._type = type
        if self._type is None:
            raise ValueError(
                f'The type of ConnectionRecorder {model} is not recognized.'
                f' Supported types: {self.CONNECTION_RECORDER_TYPES}.'
            )

    @if_not_created
    def create(self):
        """Create the ConnectionRecorder and update Connection object.

        The synapse model of the Connection object is modified so that it sends
        spikes to the ConnectionRecorder object
        """
        import nest
        # Create recorder node
        self._gid = nest.Create(self._model)
        # Update the Connection object so that it connects to the
        # ConnectionRecorder
        self._connection.connect_connection_recorder(
            recorder_type=self._type,
            recorder_gid=self._gid[0],
        )
        # Update attributes with nest.GetStatus calls
        # TODO: Update other attributes?
        self._record_to = nest.GetStatus(self.gid, 'record_to')[0]
        # Set the label (Makes the raw data filenames human-readable)
        self.set_label()
        self._filenames = self.raw_data_filenames()
        self._colnames = self.raw_data_colnames()

    def __str__(self):
        return self._model + '_' + str(self._connection)

    def get_connection_recorder_metadata_dict(self):
        metadata_dict = self.get_base_metadata_dict()
        # TODO save source and target populations' location-by-gid?
        metadata_dict.update({
            'connection_name': self._connection_name,
        })
        return metadata_dict

    def raw_data_colnames(self):
        if self._type == 'weight_recorder':
            # TODO
            return None

    def save_metadata(self, output_dir):
        metadata_path = save.output_path(
            output_dir,
            'connection_recorders_metadata',
            self._label
        )
        save.save_as_yaml(metadata_path,
                          self.get_connection_recorder_metadata_dict())
