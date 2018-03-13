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
    'dump_connection': False,
    'plot_connection': False,
    'recorders': [],
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
        # Synapse model is retrieved either from nest_params or UnitConns
        self._synapse_model = None
        # Initialize the recorders
        self.recorders = [
            ConnectionRecorder(recorder_type, recorder_params)
            for recorder_type, recorder_params in self.params['recorders']
        ]
        self.check()

    # Properties:
    @if_created
    @property
    def synapse_model(self):
        return self._synapse_model

    @property
    def dump_connection(self):
        return self.params['dump_connection']

    @property
    def plot_connection(self):
        return self.params['plot_connection']

    @property
    def scale_factor(self):
        return self.params['scale_factor']

    @property
    def save(self):
        return self.params['save']

    @property
    def save(self):
        return self.params['save']

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

    # Creation and connection

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

    def create(self):
        """Create the connections in NEST and the connection recorders."""

    def create_recorders(self):
        """Create and connect the connection recorders."""
        pass

    # Saving and dumping

    def save(self, output_dir):
        # TODO
        for field in self.save:
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
        return {driver: conns.get(driver, [])
                for driver in self.driver_gids()}

    def dump(self, output_dir):
        # TODO: Query using synapse labels to identify connections with same
        # source pop, target pop and synapse model
        if self.dump_connection:
            conns = nest.GetConnections(
                source=self.source.gids(population=self.source_population),
                target=self.target.gids(population=self.target_population),
                synapse_model=self.synapse_model)
            # We save: source_gid, target_gid, synapse_model, weight, delay
            with open(join(save.output_subdir(output_dir, 'dump'),
                           self.__str__), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(self.format_dump(conns))

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

    @staticmethod
    def format_dumped_line(line):
        return {
            'source': int(line[0]),
            'target': int(line[1]),
            'synapse_model': str(line[2]),
            'weight': float(line[3]),
            'delay': float(line[4])
        }

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
        return synapse_model

    def format_conns(self):
        """Format the self.conns() dict in a form suitable for nest.Connect."""
        # import ipdb; ipdb.set_trace()
        all_conns = list(itertools.chain(*self.conns.values()))
        sources = [conn.params['source'] for conn in all_conns]
        targets = [conn.params['target'] for conn in all_conns]
        params = {'weight': [conn.params['weight'] for conn in all_conns],
                  'delay': [conn.params['delay'] for conn in all_conns],
                  'model': self.synapse_model}
        return sources, targets, params

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
        self._synapse_model = None

    def create(self):
        # Get connections
        self.conns = self.load_conns()
        # Get synapse model from connections
        self._synapse_model = self.get_synapse_model()
        # Create connections
        self._connect()
        # Create recorders
        self.create_recorders()


class TopoConnection(BaseConnection):
    """Represent a topological connection."""

    def __init__(self, source, target, model, conn_dict):
        super().__init__(source, target, model, conn_dict)
        self._scale_factor = self.scale_factor
        self.update_nest_params()
        self._synapse_model = self.get_synapse_model()

    def update_nest_params(self):
        """Update in place self.nest_params to scale kernels and set pops."""
        # TODO: Get a view of the kernel, mask, and weights inherited from the
        # connection model

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
            'weights': self.scale_weights(self.nest_params['weights']),
        })
        # Set source populations if available
        if self.source_population:
            self.nest_params['sources'] = {'model': self.source_population}
        if self.target_population:
            self.nest_params['targets'] = {'model': self.target_population}

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

    def scale_weights(self, weights):
        # Default to no scaling
        gain = self.source.params.get('weight_gain', 1.0)
        return weights * gain

    @if_not_created
    def create(self):
        self.source._connect(self.target, self.nest_params)
        # Create recorders
        self.create_recorders()


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
        self._synapse_model = None
        # TODO: same for InputLayer connections. ( or !just.don't.care!)
        if isinstance(self.source, InputLayer):
            raise NotImplementedError

    def create(self):
        # Load and rescale connections
        self.model_conns = self.load_conns()
        self.conns = self.redraw_conns()
        # Get synapse model from connections
        self._synapse_model = self.get_synapse_model()
        # Create connections
        self._connect()

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
        self._weight = None
        self._delay = None
        self._source = None
        self._target = None

    def sort_key(self):
        return (self.synapse_model, self.source, self.target, self.weight,
                self.delay)

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()
