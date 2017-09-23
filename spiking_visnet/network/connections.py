#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/connections.py

"""Define Connections objects"""

import csv
import itertools
from collections import ChainMap
from copy import deepcopy
from os.path import join

import nest

from tqdm import tqdm

from . import topology
from .utils import NestObject, if_created, if_not_created


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
        assert self.type in ['topological', 'rescaled']
        assert self.type != 'rescaled' or self.source_dir is not None

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


    def dump(self, dump_dir):
        # TODO: Query using synapse labels to identify connections with same
        # source pop, target pop and synapse model
        if self.model.dump_connection:
            conns = nest.GetConnections(
                source=self.source.gids(population=self.source_population),
                target=self.target.gids(population=self.target_population),
                synapse_model=self.nest_params['synapse_model'])
            # We save: source_gid, target_gid, synapse_model, weight, delay
            with open(join(dump_dir, self.__str__), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(format_dump(conns))

    def save_plot(self, plot_dir):
        import matplotlib.pyplot as plt
        fig = self.plot_conn()
        plt.savefig(join(plot_dir, self.__str__))
        plt.close()

    def plot_conn(self):
        """Plot the targets of a unit using nest.topology function."""
        # TODO: Get our own version so we can plot convergent connections
        import nest.topology as tp
        fig = tp.PlotLayer(self.target.gid)
        ctr = self.source.find_center_element(population=self.source_population)
        try:
            tp.PlotTargets(ctr,
                           self.target.gid,
                           tgt_model=self.target_population,
                           syn_type=self.nest_params['synapse_model'],
                           fig=fig,
                           tgt_size=40,
                           src_size=250,
                           mask=self.nest_params['mask'],
                           kernel=self.nest_params['kernel'],
                           kernel_color='green',
                           tgt_color='yellow')
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


class RescaledConnection(Connection):

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
        # TODO: same for InputLayer connections. ( or !just.don't.care!)
        if type(self.source_layer).__name__ == 'InputLayer':
            raise NotImplementedError

    def create(self):
        print(self.__str__)
        self.model_conns = self.load_model_conns()
        self.conns = self.redraw_conns()
        self.check_conns()
        for conn in itertools.chain(*self.conns.values()):
            conn.create()

    def driver_gids(self):
        return self.driver_layer.gids(population=self.driver_population)

    def pool_gids(self):
        return self.pool_layer.gids(population=self.pool_population)

    def check_conns(self):
        driver_gids = self.driver_gids()

        for gid in driver_gids:
            nmodel = len(self.model_conns[gid])
            nrescale = len(self.conns[gid])
            assert nmodel == nrescale
            if not self.nest_params['allow_multapses']:
                assert len(set(self.conns[gid])) == len(self.conns[gid])
            if not self.nest_params['allow_autapses']:
                assert gid not in self.conns[gid]

    def load_model_conns(self):
        """Return a dictionary of model connections. Keys are driver gids."""
        model_conns = {}
        with open(join(self.model.source_dir, self.__str__), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                params = format_dumped_line(line)
                unitconn = UnitConn(params['synapse_model'], params)
                driver_gid = unitconn.params[self.driver]
                model_conns[driver_gid] = (model_conns.get(driver_gid, [])
                                           + [unitconn])
        # Return a connection list (possibly empty) for each driver gid
        return {driver: model_conns.get(driver, [])
                for driver in self.driver_gids()}

    def redraw_conns(self):
        conns = {}
        # TODO: Parallelize
        for driver in tqdm(self.driver_gids(),
                           desc=('Rescaling ' + self.__str__)):
            # Copy the model connection list
            # print('driver: ', driver, 'n_model_conns: ', len(self.model_conns.get(driver, [])))
            conns[driver] = list(self.model_conns[driver])
            # Draw the model's number of pooled gids for each driving unit
            pool_gids = self.draw_pool_gids(driver,
                                            N=len(self.model_conns[driver]))
            # Replace the model gids by the drawn gids in each UnitConn
            for i, unitconn in enumerate(conns[driver]):
                unitconn.params[self.pool] = pool_gids[i]
            # TODO: Redraw delays and weights if they have a spatial profile?
        return conns

    def draw_pool_gids(self, driver_gid, N=1):
        return topology.draw_pool_gids(self, driver_gid, N=N)


class UnitConn(NestObject):
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

    def create(self):
        import nest
        self._synapse_model = self.params['synapse_model']
        self._weight = self.params['weight']
        self._source = self.params['source']
        self._target = self.params['target']
        self._delay = self.params['delay']
        nest.Connect((self._source,),
                     (self._target,),
                     syn_spec= {'model': self._synapse_model,
                                'weight': self._weight,
                                'delay': self._delay})


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
    return formatted

def format_dumped_line(line):
    return {
        'source': int(line[0]),
        'target': int(line[1]),
        'synapse_model': str(line[2]),
        'weight': float(line[3]),
        'delay': float(line[4])
    }
