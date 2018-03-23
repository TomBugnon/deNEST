#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py
"""Connection classes."""

import csv
import itertools
from collections import ChainMap
from copy import deepcopy
from os.path import join

import nest
import numpy as np
from tqdm import tqdm

from . import topology
from .. import save
from .layers import InputLayer
from .nest_object import NestObject
from .recorders import ConnectionRecorder
from .utils import if_created, if_not_created

# Matched substrings when scaling masks.
SCALED_MASK_SUBSTRINGS = ['radius', 'lower_left', 'upper_right']

# Recognized non-float types when scaling kernels.
SCALED_KERNEL_TYPES = ['gaussian']

# List of Connection and ConnectionModel parameters (and their default values
# that shouldn't be considered as 'nest_parameters'
NON_NEST_CONNECTION_PARAMS = {
    'type': 'topological', # 'Topological', 'Rescaled' or 'FromFile'
    'source_dir': None, # Where to find the data for 'FromFile' and 'Rescaled' conns.
    'scale_factor': 1.0, # Scaling of mask and kernel
    'weight_gain': 1.0, # Scaling of synapse default weight
    'dump_connection': False,
    'plot_connection': False,
    'recorders': {},
    'save': [],
}

class ConnectionModel(NestObject):
    """Represent a NEST connection model.

    The nest parameters (`self.nest_params`) of a ConnectionModel object contain
    the base nest parameters used in Connection objects. The parameters that
    should not be considered as "nest-parameters" (listed along with their
    default values in the global variable NON_NEST_CONNECTION_PARAMS) are popped
    off the `self.nest_params` dictionary and kept in the `self.params`
    attribute.
    The population-to-population Connection objects inherit from both the params
    and the nest_params dictionaries.
    """

    def __init__(self, name, all_params):
        # Pop off the params that shouldn't be considered as NEST parameters
        nest_params = deepcopy(dict(all_params))
        params = {}
        for non_nest_param, default in NON_NEST_CONNECTION_PARAMS.items():
            params[non_nest_param] = nest_params.pop(non_nest_param, default)
        # Check that there is no 'weight' parameter (refactor `weight` in
        # `weight_gain`)
        error_msg = ("`weights` is not an acceptable parameter. Please use the"
                     " parameter `weight_gain` which will scale the synapse"
                     " default.")
        assert ('weights' not in nest_params), error_msg
        # We now save the params and nest_params dictionaries as attributes
        super().__init__(name, params)
        self.nest_params = nest_params
        # Check that the connection types are recognized and nothing is missing.
        assert self.type in ['topological', 'rescaled', 'from_file']
        assert self.type != 'rescaled' or self.source_dir is not None
        assert self.type != 'from_file' or self.source_dir is not None

    @property
    def type(self):
        return self.params['type']

    @property
    def source_dir(self):
        return self.params['source_dir']

class BaseConnection(NestObject):
    """Base class for all population-to-population connections.

    A Connection consists in synapses between two populations that have a
    specific synapse model.

    Population-to-population connections are described by a dictionnary of the
    following form::
    {
        `source_layer`: 'source_layer',
        `source_population`: 'source_population',
        `target_layer`: 'target_layer',
        `target_population`: 'target_population',
        `model`: 'connection_model'
        `params`: 'non-nest-parameters',
        `nest_params`: 'nest_params',
    }

    A Connection's `nest_params` and `params` are inherited and ChainMapped from
    its ConnectionModel model.
    The "non-nest" parameters (listed along with their default values in the
    `NON_NEST_CONNECTION_PARAMS`) are popped off the `nest_params` parameters
    at initialization and creation
    Connections() inherit and possibly override their parameters (using a
    ChainMap) from their respective ConnectionModel model. From their merged
    connection dictionary, certain parameters are popped off and saved as
    attributes. The remaining parameters are NEST parameters that are passed to
    the kernel during a `Connect()` or `ConnectLayers()` call.
    The parameters that shouldn't be considered as NEST parameters (and should
    therefore be removed from the parameters during initialization or creation)
    are listed in the global variable `NON_NEST_CONNECTION_PARAMS`.

    Connection weights can be recorded by 'weight_recorder' devices. Because
    the weight recorder device's GID must be specified in a synapse model's
    default parameters (using nest.SetDefaults or nest.CopyModel), the actual
    NEST synapse model used when connecting might differ from the one specified
    in the network's synapse models.

    The workflow for creating connections and their respective recorders is as
    follows:
        1- Initialize the Connection object and possibly its Recorder object
        2- Create the Recorder object.
        3- Get the GID of the Recorder object and create a new NEST synapse
            model that will send the spikes to the recorder. The name of the
            synapse model is saved in self.nest_synapse_model.
        4- Create the connection with the self.nest_synapse_model model.
    self.nest_synapse_model is only used to communicate with the kernel. The
    Connection is still denoted by its source and target population and its
    "base" synapse model (saved in self.synapse_model)
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, source, target, model, connection_dict):
        """Initialize Connection object from model and overrides.

        Initialize the self.params and self.nest_params attributes, and all the
        other attributes as well :)

        Args:
            source (Layer): source Layer object
            target (Layer): target Layer object
            model (ConnectionModel): ConnectionModel object. Provide base
                'params' and 'nest_params' parameter dictionaries.
            connection_dict (dict): Dictionary defining the connection. The
                dictionary should have the form described in the class
                docstring. In particular, it may contain the following keys:
                    params (dict): "non-nest" parameter dictionary. Combined in
                        a ChainMap with `model.params`. All recognized
                        parameters are listed in global variable
                        `NON_NEST_CONNECTION_PARAMS`.
                    nest_params (dict): Parameters that may be passed to the
                        NEST kernel. Combined in a ChainMap with
                        model.nest_params. No parameter listed in global
                        variable `NON_NEST_CONNECTION_PARAMS` should be present
                        in this variable.
        """
        ##
        # Check the params and nest_params dictionaries and ChainMap them with
        # the ConnectionModel params and nest_params
        params = connection_dict.get('params', {})
        nest_params = connection_dict.get('nest_params', {})
        assert all([key in NON_NEST_CONNECTION_PARAMS for key in
                    params.keys()])
        assert not any([key in NON_NEST_CONNECTION_PARAMS for key in
                        nest_params.keys()])
        self.params = dict(ChainMap(params, model.params))
        self.nest_params = dict(ChainMap(nest_params, model.nest_params))
        super().__init__(model.name, self.params)
        ##
        # Define the source/target population attributes
        self.model = model
        self.source = source
        self.source_population = connection_dict.get('source_population', None)
        self.target = target
        self.target_population = connection_dict.get('target_population', None)
        # By default, we consider the driver to be the source
        self.driver = 'source'
        self.driver_layer = self.source
        self.driver_population = self.source_population
        ##
        # Synapse model is retrieved either from nest_params or UnitConns.
        # The synapse model used in NEST might have a different name since we
        # need to change the default parameters of a synapse to specify the
        # weight recorder
        self._synapse_model = None
        self._nest_synapse_model = None
        # Initialize the recorders
        self.recorders = [
            ConnectionRecorder(recorder_name, recorder_params)
            for recorder_name, recorder_params
            in self.params['recorders'].items()
        ]
        assert len(self.recorders) < 2 # Only a single recorder type so far...
        self.check()

    # Properties:
    @property
    def synapse_model(self):
        return self._synapse_model

    @property
    def nest_synapse_model(self):
        return self._nest_synapse_model

    @property
    def dump_connection(self):
        return self.params['dump_connection']

    @property
    def plot_connection(self):
        return self.params['plot_connection']

    @property
    def scale_factor(self):
        return self._scale_factor

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

    def driver_gids(self):
        return self.driver_layer.gids(population=self.driver_population)

    def pool_gids(self):
        return self.pool_layer.gids(population=self.pool_population)

    def source_gids(self):
        return self.source.gids(population=self.source_population)

    def target_gids(self):
        return self.target.gids(population=self.target_population)

    # Query stuff

    def get_recorders(self, recorder_type=None):
        for recorder in self.recorders:
            if recorder_type is None or recorder.type == recorder_type:
                yield recorder

    # Creation and connection

    def create(self):
        """Create the connections in NEST and the connection recorders.

        Should use in order the following steps:
            1- create recorders
            2- create nest_synapse_model
            3- connect
        """
        # Get the UnitConns (does nothing for topological conns)
        self.get_connections()
        # Get the base synapse model from the list of UnitConns or the
        # nest_params
        self.get_synapse_model()
        # Create recorder objects
        self.create_recorders()
        # Get the NEST synapse model (different from synapse model if we record
        # the connection
        self.create_nest_synapse_model()
        # Update the nest_parameters to get the proper connection weight, set
        # the proper nest_synapse_model, possibly rescale kernels and weights,
        # etc
        self.update_nest_params()
        # Actually create the connections in NEST
        self._connect()

    def get_connections(self):
        """Get the UnitConns from file (for Rescaled and FromFile conns."""
        pass

    def get_synapse_model(self):
        """Get synapse model either from nest params or conns list."""
        # Nasty hack to access the synapse model for FromFileConnections
        synapse_model = None
        try:
            synapse_model = self.nest_params['synapse_model']
        except (AttributeError, KeyError):
            # Get the synapse model of any UnitConn.
            for conn in self.conns.values():
                if conn:
                    synapse_model = conn[0].params['synapse_model']
                    break
        if synapse_model is None:
            #TODO
            import warnings
            warnings.warn('Could not determine synapse model since there was'
                            ' no dumped connection')
        self._synapse_model = synapse_model

    def create_recorders(self):
        """Create and connect the connection recorders."""
        conn_params = {
            "connection_name": self.__str__,
            "src_layer_name": self.source.name,
            "src_population_name": self.source_population,
            "src_gids": self.source_gids(),
            "tgt_layer_name": self.target.name,
            "tgt_population_name": self.target_population,
            "tgt_gids": self.target_gids(),
            "synapse_model": self.synapse_model,
        }
        for recorder in self.recorders:
            recorder.create(conn_params)

    def create_nest_synapse_model(self):
        """Create a new synapse model that sends spikes to the recorder."""
        import nest
        if not self.recorders:
            self._nest_synapse_model = self.synapse_model
        else:
            recorder = self.recorders[0]
            assert recorder.type == 'weight_recorder'
            self._nest_synapse_model = self.nest_synapse_model_name()
            nest.CopyModel(self.synapse_model, self._nest_synapse_model,
                           {
                               recorder.type: recorder.gid[0]
                           })

    def nest_synapse_model_name(self):
        return f"{self.synapse_model}-{self.__str__}"

    def update_nest_params(self):
        pass

    def _connect(self):
        """Call nest.Connect() to create all unit connections.

        We call nest.Connect() with the following arguments:
            <sources> (list): list of gids.
            <targets> (list): list of gids.
            conn_spec='one_to_one'
            syn_spec=params
        where params is of the form::
            {
                <weight>: <list_of_weights>
                <delays>: <list_of_delays>
                <model>: <synapse_model>
            }
        """
        sources, targets, params = self.format_conns()
        nest.Connect(sources, targets,
                     conn_spec='one_to_one',
                     syn_spec=params)

    def format_conns(self):
        """Format the self.conns() dict in a form suitable for nest.Connect."""
        all_conns = list(itertools.chain(*self.conns.values()))
        sources = [conn.params['source'] for conn in all_conns]
        targets = [conn.params['target'] for conn in all_conns]
        params = {'weight': [conn.params['weight'] for conn in all_conns],
                  'delay': [conn.params['delay'] for conn in all_conns],
                  'model': self.nest_synapse_model}
        return sources, targets, params

    # Connection loading from file

    def load_conns(self):
        """Return a dictionary of model connections. Keys are driver gids.

        The returned dictionary is of the form::
            {
                <driver_gid>: <UnitConn_list>
            }
        Where <UnitConn_list> is a list of UnitConn single connections for the
        considered driver unit.

        NB: By default (ie, if the connection is not topological) the driver
        layer is considered to be the source.
        """
        conns = {}
        with open(join(self.model.source_dir, self.__str__), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                params = self.format_dumped_line(line)
                unitconn = UnitConn(params['synapse_model'], params)
                driver_gid = unitconn.params[self.driver]
                conns[driver_gid] = (conns.get(driver_gid, [])
                                           + [unitconn])
        # Return a connection list (possibly empty) for each driver gid
        import warnings
        warnings.warn('Double check the synapse models vs nest_synapse_model')
        return {driver: conns.get(driver, [])
                for driver in self.driver_gids()}

    @staticmethod
    def format_dumped_line(line):
        return {
            'source': int(line[0]),
            'target': int(line[1]),
            'synapse_model': str(line[2]),
            'weight': float(line[3]),
            'delay': float(line[4])
        }

    # Connection dumping  to file

    def dump(self, output_dir):
        # TODO: Query using synapse labels to identify connections with same
        # source pop, target pop and synapse model
        if self.dump_connection:
            conns = nest.GetConnections(
                source=self.source.gids(population=self.source_population),
                target=self.target.gids(population=self.target_population),
                synapse_model=self.nest_synapse_model)
            # We save: source_gid, target_gid, synapse_model, weight, delay
            with open(join(save.output_subdir(output_dir, 'dump'),
                           self.__str__), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(self.format_dump(conns))
        import warnings
        warnings.warn('Double check the synapse models vs nest_synapse_model')

    @staticmethod
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
        return sorted(formatted)

    # Save and plot stuff

    def save(self, output_dir):
        self.save_recorders(output_dir)
        self.save_synapse_state(output_dir)

    def save_recorders(self, output_dir):
        """Save recorders' data."""
        for recorder in self.recorders:
            recorder.save(output_dir)

    def save_synapse_state(self, output_dir):
        """Save using a GetConnections() call."""
        for field in self.params['save']:
            # TODO
            print('TODO: save connection ', field, ' in ', output_dir)


    def save_plot(self, output_dir):
        if self.plot_connection:
            import matplotlib.pyplot as plt
            fig = self.plot_conn() #pylint: disable=unused-variable
            plt.savefig(join(save.output_subdir(output_dir, 'connections'),
                             self.__str__))
            plt.close()

    def plot_conn(self):
        """Plot the targets of a unit using nest.topology function."""
        # TODO: Get our own version so we can plot convergent connections
        import nest.topology as tp
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        tp.PlotLayer(self.target.gid, fig)
        ctr = self.source.find_center_element(population=self.source_population)
        # Plot the kernel and mask if the connection is topological or rescaled.
        try:
            tp.PlotKernel(ax, ctr,
                          self.nest_params['mask'],
                          kern=self.nest_params['kernel'],
                          kernel_color='green')
        except (AttributeError, KeyError, ValueError):
            # AttributeError, KeyError: if no nest_params mask or kernel
            # ValueError: if the mask or kernel cannot be plotted (custom mask)
            pass
        try:
            tp.PlotTargets(ctr,
                           self.target.gid,
                           tgt_model=self.target_population,
                           syn_type=self.synapse_model,
                           fig=fig,
                           tgt_size=40,
                           src_size=250,
                           tgt_color='red')
        except ValueError:
            print((f"Not plotting targets: the center unit {ctr[0]} has no "
                    + f"target within connection {self.__str__}"))
        plt.title(f"Plot of targets of a single source unit.\n"
                  f"Target units' pop: {self.target.name},"
                  f"{str(self.target_population)} (targets in red),\n"
                  f"Source unit's population: {self.source.name},"
                  f"{str(self.source_population)}\n"
                  f"Connection name: {self.name},\n", fontsize=7)
        footnote = ("NB: The actual connection probability might be smaller "
                    "than it seems if there is multiple units per grid position"
                    " in the target population(s)")
        ax.annotate(footnote, xy=(1, 0), xycoords='axes fraction', fontsize=5,
            xytext=(0, -15), textcoords='offset points',
            ha='right', va='top')
        return fig

    def check(self):
        """Check the connection to avoid bad errors.

        Make sure that:
        - the target is not an ``InputLayer``,
        - if the source is an ``InputLayer``, the source population is a parrot
        neuron (otherwise we can't record the input layer).
        """
        assert(type(self.target).__name__ != 'InputLayer')
        if (type(self.source).__name__ == 'InputLayer'
            and self.source_population != self.source.PARROT_MODEL):
                import warnings
                warn_str = (f'\n\nCareful! The Input population for connection:'
                            f'\n{self.__str__}\n is not a parrot '
                            'neuron! This might throw a bad NEST error.\n\n\n')
                warnings.warn(warn_str)


class FromFileConnection(BaseConnection):
    """Represent a connection loaded from file."""

    def __init__(self, source, target, model, params):
        super().__init__(source, target, model, params)
        self.conns = None

    # Creation functions not inherited from BaseConnection

    def get_connections(self):
        self.conns = self.load_conns()


class TopoConnection(BaseConnection):
    """Represent a topological connection."""

    def __init__(self, source, target, model, conn_dict):
        super().__init__(source, target, model, conn_dict)
        self._scale_factor = self.params['scale_factor']

    # Creation functions not inherited from BaseConnection

    def get_connections(self):
        pass

    def update_nest_params(self):
        """Update in place self.nest_params.

        - Set source and target populations,
        - Set NEST synapse model (possibly different from self.synapse_model)
        - Set connection weight: Scale synapse default by Connection and Layer
            `weight_gain` params
        - Scale kernels,
        - Scale masks,
        """
        # TODO: Get a view of the kernel, mask, and weights inherited from the
        # connection model
        self.set_populations_nest_params()
        self.set_synapse_model_nest_params()
        self.set_connection_weight()
        self.scale_kernel_mask()

    def set_synapse_model_nest_params(self):
        """Update the synapse_model given to NEST."""
        self.nest_params['synapse_model'] = self.nest_synapse_model

    def set_populations_nest_params(self):
        """Set the source and target populations in self.nest_params."""
        if self.source_population:
            self.nest_params['sources'] = {'model': self.source_population}
        if self.target_population:
            self.nest_params['targets'] = {'model': self.target_population}

    def set_connection_weight(self):
        """Set connection weight in nest_params from synapse default.

        The Connection's weight is equal to the synapse model's default weight,
        scaled by the Connection's `weight_gain` parameter, and by the source
        layer's `weight_gain` parameter.
        """
        synapse_df_weight = nest.GetDefaults(self.nest_params['synapse_model'],
                                             'weight')
        scaled_weight = self.scale_weights(synapse_df_weight)
        if synapse_df_weight != 1.0:
            print(f'NB: Connection weights scale synapse default:'
                  f'{self.__str__}: weights = {scaled_weight}')
        self.nest_params['weights'] = scaled_weight

    def scale_weights(self, weights):
        """Scale the synapse weight by Connection and Layer's weight gain."""
        connection_gain = self.params['weight_gain']
        layer_gain = self.source.params.get('weight_gain', 1.0)
        return weights * connection_gain * layer_gain

    def scale_kernel_mask(self):
        """Update self._scale_factor and scale kernels and masks."""

        # Get connection-specific scaling factor, taking in account whether the
        # connection is convergent or divergent
        if (self.nest_params['connection_type'] == 'convergent'
                and self.source.params.get('scale_kernels_masks_to_extent', True)):
            # For convergent connections, the pooling layer is the source
            self._scale_factor = self.source.extent_units(self.scale_factor)
        elif (self.nest_params['connection_type'] == 'divergent'
                and self.target.params.get('scale_kernels_masks_to_extent', True)):
            # For convergent connections, the pooling layer is the target
            self._scale_factor = self.target.extent_units(self.scale_factor)

        # Set kernel, mask, and weights, scaling if necessary
        self.nest_params.update({
            'kernel': self.scale_kernel(self.nest_params.get('kernel', {})),
            'mask': self.scale_mask(self.nest_params.get('mask', {})),
        })

    def scale_kernel(self, kernel):
        """Return a new kernel scaled by ``scale_factor``.

        If kernel is a float: copy and return
        If kernel is a dictionary:
            - If empty: return ``{}``
            - If recognized type (in SCALED_KERNEL_TYPES): scale 'sigma' field
                and return scaled kernel
            - If unrecognized type (not in SCALED_KERNEL_TYPES): issue warning
                and return same kernel
        """
        kernel = deepcopy(kernel)
        try:
            # Float kernel (not scaled)
            return float(kernel)
        except TypeError:
            if not kernel:
                # Empty kernel
                return kernel
            kernel_type = list(kernel.keys())[0]
            if kernel_type in SCALED_KERNEL_TYPES:
                # Recognized non-trivial kernel type (scale sigma parameter)
                kernel[kernel_type]['sigma'] *= self.scale_factor
                return kernel
            # Unrecognized non-trivial kernel type (return and warn)
            import warnings
            warnings.warn('Not scaling unrecognized kernel type')
            return kernel

    def scale_mask(self, mask):
        """Return a new mask scaled by ``scale_factor``.

        Scale the fields of the mask parameters if their key contains the
        strings 'radius' (for circular and doughnut mask.), 'lower_left' or
        'upper_right' (for rectangular or box masks.)
        You can modify the list of scaled field by changing the
        ``SCALED_MASK_SUBSTRINGS`` constant.

        Return ``{}`` if ``mask`` is empty.
        """
        mask = deepcopy(mask)
        mask_type, mask_params = list(mask.items())[0]
        # Iterate on all the parameters of the mask
        for key in mask_params:
            # Test if the key contains any of the matching substrings.
            if any([substring in key for substring
                    in SCALED_MASK_SUBSTRINGS]):
                try:
                    # If entry is list, scale all the elements
                    mask[mask_type][key] = [scalar * self.scale_factor
                                            for scalar
                                            in mask[mask_type][key]]
                except TypeError:
                    # If entry is float, scale it
                    mask[mask_type][key] *= self.scale_factor
        return mask

    def _connect(self):
        self.source._connect(self.target, self.nest_params)


class RescaledConnection(TopoConnection):
    """Represent a rescaled topological connection from a dump."""

    def __init__(self, source, target, model, conn_dict):
        super().__init__(source, target, model, conn_dict)
        if self.nest_params['connection_type'] == 'convergent':
            self.pool = 'source'
            self.pool_layer = self.source
            self.pool_population = self.source_population
            self.driver = 'target'
            self.driver_layer = self.target
            self.driver_population = self.target_population
        elif self.nest_params['connection_type'] == 'divergent':
            self.pool = 'target'
            self.pool_layer = self.target
            self.pool_population = self.target_population
            self.driver = 'source'
            self.driver_layer = self.source
            self.driver_population = self.source_population
        # Both are dictionaries: {'driver_gid': [UnitConn, ...]}
        self.model_conns = None
        self.conns = None
        # TODO: same for InputLayer connections. ( or !just.don't.care!)
        if isinstance(self.source, InputLayer):
            raise NotImplementedError

    # Creation functions not inherited from BaseConnection

    def get_connections(self):
        # Load and rescale connections
        self.model_conns = self.load_conns()
        self.conns = self.redraw_conns()

    def redraw_conns(self):
        """Redraw pool gids according to topological parameters."""
        PARALLEL = True
        drivers = self.driver_gids()
        # TODO: Parallelize
        # Draw the model's number of pooled gids for each driving unit
        if PARALLEL:
            from joblib import Parallel, delayed
            print('Rescaling ', self.__str__)
            arg_list = [(driver, len(self.model_conns[driver]))
                        for driver in drivers]
            all_pool_gids = Parallel(n_jobs=8, verbose=1)(
                delayed(self.draw_pool_gids)(*args) for args in arg_list
            )
        else:
            all_pool_gids = [
                self.draw_pool_gids(driver, N=len(self.model_conns[driver]))
                for driver in tqdm(drivers, desc=('Rescaling ' + self.__str__))
            ]
        # Copy the model connection list
        conns = deepcopy(self.model_conns)
        # Replace the model gids by the drawn gids in each UnitConn
        for driver, pool_gids in zip(drivers, all_pool_gids):
            for i, unitconn in enumerate(conns[driver]):
                unitconn.params[self.pool] = pool_gids[i]
                # TODO: Redraw delays and weights if they have a spatial
                # profile?
        return conns

    def draw_pool_gids(self, driver_gid, N=1):
        return topology.draw_pool_gids(self, driver_gid, N=N)


class UnitConn(NestObject):
    """Represent a single connection between two neurons."""

    def __init__(self, name, params):
        super().__init__(name, params)
        self._synapse_model = None
        #TODO Check that we use the nest_synapse_model and not synapse_model
        self._nest_synapse_model = None
        self._weight = None
        self._delay = None
        self._source = None
        self._target = None

    def sort_key(self):
        return (self.synapse_model, self.source, self.target, self.weight,
                self.delay)

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()
