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


class ConnectionModel(NestObject):
    """Represent a NEST connection model."""
    DEFAULT_SCALE_FACTOR = 1.0

    def __init__(self, name, params):
        super().__init__(name, params)
        self._scale_factor = self.params.pop('scale_factor',
                                             self.DEFAULT_SCALE_FACTOR)
        self._type = self.params.pop('type', 'topological')
        self._source_dir = self.params.pop('source_dir', None)
        self._dump_connection = self.params.pop('dump_connection', False)
        self._plot_connection = self.params.pop('plot_connection', True)
        assert self.type in ['topological', 'rescaled', 'from_file']
        assert self.type != 'rescaled' or self.source_dir is not None
        assert self.type != 'from_file' or self.source_dir is not None

    @property
    def type(self):
        return self._type

    @property
    def dump_connection(self):
        return self._dump_connection

    @property
    def source_dir(self):
        return self._source_dir

    @property
    def scale_factor(self):
        return self._scale_factor

class BaseConnection(NestObject):
    """Base class for all connection types."""
    def __init__(self, source, target, model, params):
        super().__init__(model.name, params)
        self.model = model
        self.source = source
        self.source_population = params.get('source_population', None)
        self.target = target
        self.target_population = params.get('target_population', None)
        # Synapse model is retrieved either from nest_params or UnitConns
        self.synapse_model = None
        # By default, we consider the driver to be the source
        self.driver = 'source'
        self.driver_layer = self.source
        self.driver_population = self.source_population

    def save(self, output_dir):
        # TODO
        for field in self.params.get('save', []):
            print('TODO: save connection ', field, ' in ', output_dir)

    def save_plot(self, output_dir):
        if self.model._plot_connection:
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
        except (AttributeError, KeyError):
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
        return fig

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
        if self.model.dump_connection:
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
        # import ipdb; ipdb.set_trace()
        all_conns = list(itertools.chain(*self.conns.values()))
        sources = [conn.params['source'] for conn in all_conns]
        targets = [conn.params['target'] for conn in all_conns]
        params = {'weight': [conn.params['weight'] for conn in all_conns],
                  'delay': [conn.params['delay'] for conn in all_conns],
                  'model': self.synapse_model}
        return sources, targets, params


class FromFileConnection(BaseConnection):
    """Represent a connection loaded from file."""

    def __init__(self, source, target, model, params):
        super().__init__(source, target, model, params)
        self.conns = None
        self.synapse_model = None

    def create(self):
        # Get connections
        self.conns = self.load_conns()
        # Get synapse model from connections
        self.synapse_model = self.get_synapse_model()
        # Create connections
        self._connect()


class TopoConnection(BaseConnection):
    """Represent a topological connection."""

    def __init__(self, source, target, model, params):
        super().__init__(source, target, model, params)
        self.scale_factor = None
        self.nest_params = self.get_nest_params()
        self.synapse_model = self.get_synapse_model()

    def get_nest_params(self):
        # Get NEST connection parameters for a topological connection
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


class RescaledConnection(TopoConnection):
    """Represent a rescaled topological connection from a dump."""

    def __init__(self, source, target, model, params):
        super().__init__(source, target, model, params)
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
        self.synapse_model = None
        # TODO: same for InputLayer connections. ( or !just.don't.care!)
        if isinstance(self.source, InputLayer):
            raise NotImplementedError

    def create(self):
        # Load and rescale connections
        self.model_conns = self.load_conns()
        self.conns = self.redraw_conns()
        # Get synapse model from connections
        self.synapse_model = self.get_synapse_model()
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
