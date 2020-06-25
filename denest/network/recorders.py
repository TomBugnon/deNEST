#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# recorders.py

"""PopulationRecorder and ProjectionRecorder objects."""

import logging

from ..base_object import NestObject
from ..io import save
from .utils import if_created, if_not_created

log = logging.getLogger(__name__)


# Parameters that are recognized and not passed as NEST parameters. These
# parameters are set in `populations.yml` rather than `recorders.yml`
NON_NEST_PARAMS = {}


class BaseRecorder(NestObject):
    """Base class for all recorder classes. Represent nodes (not models).

    Args:
        model (str): Model of the recorder in NEST. This should be a native
            model in NEST, or a recorder model defined via ``recorder_models``
            network parameters.
    """

    def __init__(self, model):
        super().__init__(model, {}, {})
        self._model = model
        self._type = None
        # Attributes below may depend on NEST default and recorder models and
        # are updated after creation
        self._gid = None  # gid of recorder node
        self._filenames = None
        self._record_to = None  # eg ['memory', 'file']
        self._withtime = None
        self._label = None  # Only affects raw data filenames.

    @property
    def model(self):
        """Return NEST model of recorder node."""
        return self._model

    @property
    @if_created
    def gid(self):
        """Return gid of node as tuple."""
        return self._gid

    @property
    @if_created
    def variables(self):
        """Return recorder's `record_from`. "spikes" for spike detectors."""
        return self._record_from

    @property
    def type(self):
        """Return type of recorder ('spike_detector', 'multimeter', ...)"""
        return self._type

    def __str__(self):
        raise NotImplementedError

    def set_status(self, nest_params):
        """Call nest.SetStatus on node."""
        import nest

        log.info(f"  Setting status for recorder %s: %s", str(self), nest_params)
        nest.SetStatus(self.gid, nest_params)

    def set_label(self):
        """Set self._label and node's NEST ``label`` from self.__str__."""
        import nest

        self._label = self.__str__()
        # Don't use self.set_status to avoid verbose
        # self.set_status({'label': self._label})
        nest.SetStatus(self.gid, {"label": self._label})

    def raw_data_colnames(self):
        """Return column names of raw data files for pandas loading."""
        raise NotImplementedError

    def raw_data_filenames(self):
        """Return filenames of raw data files saved by NEST.

        From NEST documentation::

            ```The name of the output file is
            `data_path/data_prefix(label|model_name)-gid-vp.file_extension`

            See /label and /file_extension for how to change the name.
            /data_prefix is changed in the root node.```

        NB: We don't use the recorder's `filenames` key, since it is created
        only after the first `Simulate` call.

        NB: There is one file per virtual process. The virtual processes are
        numeroted from 0 and formatted with the same number of digits as that of
        the number of virtual processes.
        """
        import nest

        if "file" not in self._record_to:
            return []
        assert self._label is not None  # Check that the label has been set
        prefix = nest.GetKernelStatus("data_prefix") + self._label + f"-{self.gid[0]}-"
        extension = nest.GetStatus(self.gid, "file_extension")[0]
        n_vp = nest.GetKernelStatus("local_num_threads")
        # TODO: CHeck the formatting for 3 digits number of threads
        assert n_vp <= 100
        n_digits = len(str(n_vp))
        return [prefix + f"{str(vp).zfill(n_digits)}.{extension}" for vp in range(n_vp)]

    @if_created
    def get_base_metadata_dict(self):
        """Return metadata dict common to all recorder types."""
        return {
            "type": self._type,
            "label": self._label,
            "colnames": self.raw_data_colnames(),
            "filenames": self.raw_data_filenames(),
        }

    def save_metadata(self):
        """Save metadata for recorder."""
        raise NotImplementedError


class PopulationRecorder(BaseRecorder):
    """Represent a recorder node. Connects to a single population.

    Handles creating and connecting the recorder node to the recorder
    population, and saving the recorder's output metadata, which contains
    population and layer information (population shape, unit locations, etc).

    PopulationRecorder objects are specified in the `population_recorders`
    parameters at network/recorders. Recorder models can be specified separately
    in the `recorder_models` parameters at network/recorders.

    Args:
        model (str): Model of the recorder in NEST. This should be a native
            model in NEST, or a recorder model defined via ``recorder_models``
            network parameters. (eg: 'multimeter' or 'modified_multimeter')
        layer (Layer): Layer object
        population_name (str): Name of population to connect to.
    """

    POP_RECORDER_TYPES = ["multimeter", "spike_detector"]

    def __init__(self, model, layer, population_name):
        """Initialize PopulationRecorder object."""
        super().__init__(model)
        self.layer = layer
        self._population_name = population_name  # Name of recorded population
        self._layer_name = self.layer.name  # Name of recorded pop's layer
        self._layer_shape = self.layer.shape  # (nrows, cols) for recorded pop
        self._population_shape = self.layer.population_shapes[
            self._population_name
        ]
        # Number of nodes per grid position for pop
        self._units_number = self.layer.populations[population_name]
        # Attributes below are necessary for creating the output metadata file
        # and are defined during the `self.create()` call.
        self._gids = None  # all gids of recorded nodes
        self._locations = None  # location by gid for recorded nodes
        # We try to guess the recorder type ('spike_detector', 'multimeter')
        # from its model name
        self._type = None  # 'spike detector' or 'multimeter'
        for rec_type in self.POP_RECORDER_TYPES:
            if rec_type in self.model:
                self._type = rec_type
                break
        if self._type is None:
            raise ValueError(
                f"The type of recorder {self.model} couldn't be guessed."
                f" Recognized types: {self.POP_RECORDER_TYPES}"
            )
        # Attributes below may depend on NEST default and recorder models and
        # are updated after creation
        self._record_from = None  # list of variables for mm, or ['spikes']
        self._interval = None  # Sampling interval. Ignored for spike_detector.

    def __str__(self):
        return self.model + "_" + self._layer_name + "_" + self._population_name

    @property
    @if_created
    def gids(self):
        """Return list of GIDs of recorded units."""
        return self._gids

    @property
    @if_created
    def locations(self):
        """Return ``{<gid>: (<row>, <col>, <unit_i>)}`` for recorded units."""
        return self._locations

    @property
    def population_name(self):
        """Return name of population the recorder is connected to."""
        return self._population_name

    @property
    def layer_name(self):
        """Return name of layer the recorder is connected to."""
        return self._layer_name

    @if_not_created
    def create(self):
        """Create the PopulationRecorder node

            1. Create the recorder node in NEST
            2. Update PopulationRecorder's attributes from the Layer object and
                the node in NEST
            3. Connect the node to the target GIDs.
        """
        import nest

        # Create node
        self._gid = nest.Create(self.model, params={})
        # Save population and layer-wide attributes
        self._gids = self.layer.gids(population=self.population_name)
        self._locations = None  # TODO
        # Update attributes after creation (may depend on nest defaults and
        # recorder models)
        self._record_to = nest.GetStatus(self.gid, "record_to")[0]
        self._withtime = nest.GetStatus(self.gid, "withtime")[0]
        if self.type == "multimeter":
            self._record_from = [
                str(variable) for variable in nest.GetStatus(self.gid, "record_from")[0]
            ]
            self._interval = nest.GetStatus(self.gid, "interval")[0]
        elif self.type == "spike_detector":
            self._record_from = ["spikes"]
        # Set the label and filename prefix
        self.set_label()
        self._filenames = self.raw_data_filenames
        # Connect population
        if self.type == "multimeter":
            nest.Connect(self.gid, self.gids)
        elif self.type == "spike_detector":
            nest.Connect(self.gids, self.gid)

    def get_population_recorder_metadata_dict(self):
        """Create population recorder metadata dict."""
        metadata_dict = self.get_base_metadata_dict()
        metadata_dict.update(
            {
                "gids": self._gids,
                "locations": self._locations,
                "population_name": self._population_name,
                "layer_name": self._layer_name,
                "layer_shape": self._layer_shape,
                "population_shape": self._population_shape,
                "units_number": self._units_number,
                "interval": self._interval,
                "record_from": self._record_from,
            }
        )
        return metadata_dict

    def save_metadata(self, output_dir):
        """Save population recorder metadata."""
        metadata_path = save.output_path(output_dir, "recorders_metadata", self._label)
        save.save_as_yaml(metadata_path, self.get_population_recorder_metadata_dict())

    def raw_data_colnames(self):
        """Return list of labels for columns in raw data saved by NEST."""
        import nest

        # TODO: Make sure this is necessary or find a workaround?
        # TODO: check the parameters at creation rather than here
        assert nest.GetStatus(self.gid, "withtime")[0]
        if self.type == "multimeter":
            return ["gid", "time"] + self._record_from
        elif self.type == "spike_detector":
            return ["gid", "time"]
        else:
            raise Exception


class ProjectionRecorder(BaseRecorder):
    """Represents a recorder connected to synapses of a ``Projection``.

    ProjectionRecorders are connected to synapses of a single
    population-to-population ``Projection`` object.

    Args:
        model (str): Model of the projection recorder in NEST. This should be a
            native model in NEST, or a recorder model defined via
            ``recorder_models`` network parameters. (eg: 'weight_recorder')
        projection (``Projection``): ``Projection`` object the recorder is
            connected to.
    """

    # pylint:disable=too-many-instance-attributes
    CONNECTION_RECORDER_TYPES = ["weight_recorder"]

    def __init__(self, model, projection):
        super().__init__(model)
        self._model = model
        self._projection = projection
        self._projection_name = str(projection)
        # "type" or ProjectionRecorder ("weight_recorder")
        self._type = None
        for type in self.CONNECTION_RECORDER_TYPES:
            if type in self._model:
                self._type = type
        if self._type is None:
            raise ValueError(
                f"The type of ProjectionRecorder {model} is not recognized."
                f" Supported types: {self.CONNECTION_RECORDER_TYPES}."
            )

    def __str__(self):
        return self._model + "_" + str(self._projection)

    @if_not_created
    def create(self):
        """Create the ProjectionRecorder and update Projection object.

            1. Create the recorder node in NEST.
            2. Modify the synapse model of the Projection object is modified so
                that it sends spikes to the ProjectionRecorder object.
            3. Update ProjectionRecorder's attributes.
        """
        import nest

        # Create recorder node
        self._gid = nest.Create(self._model)
        # Update the Projection object so that it connects to the
        # ProjectionRecorder
        self._projection._connect_projection_recorder(
            recorder_type=self._type, recorder_gid=self._gid[0],
        )
        # Update attributes with nest.GetStatus calls
        # TODO: Update other attributes?
        self._record_to = nest.GetStatus(self.gid, "record_to")[0]
        # Set the label (Makes the raw data filenames human-readable)
        self.set_label()
        self._filenames = self.raw_data_filenames()
        self._colnames = self.raw_data_colnames()

    # TODO: Save more information
    def get_projection_recorder_metadata_dict(self):
        """Create metadata directory for ProjectionRecorder."""
        metadata_dict = self.get_base_metadata_dict()
        metadata_dict.update(
            {"projection_name": self._projection_name,}
        )
        return metadata_dict

    def raw_data_colnames(self):
        """Return column names of raw data files for pandas loading."""
        if self._type == "weight_recorder":
            # TODO
            return None

    def save_metadata(self, output_dir):
        """Save recorder metadata."""
        metadata_path = save.output_path(output_dir, "recorders_metadata", self._label)
        save.save_as_yaml(metadata_path, self.get_projection_recorder_metadata_dict())
