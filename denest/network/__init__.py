#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/__init__.py

"""Provide a class to construct a network."""

import itertools
import logging

from tqdm import tqdm

from ..parameters import ParamsTree
from ..utils import validation
from ..utils.validation import ParameterError
from .projections import ProjectionModel, TopoProjection
from .layers import InputLayer, Layer
from .models import Model, SynapseModel
from .recorders import ProjectionRecorder, PopulationRecorder
from .utils import if_not_created, log

log = logging.getLogger(__name__)


LAYER_TYPES = {
    None: Layer,
    'InputLayer': InputLayer,
}

CONNECTION_TYPES = {
    'topological': TopoProjection,
}


class Network(object):
    """Represent a full network.

    Args:
        tree (ParamsTree): "network" parameter tree. The following children
            are expected:

            ``neuron_models`` (:class:`ParamsTree`)
              Parameter tree, the leaves of which define neuron models.
              Each leaf is used to initialize a :class:`Model` object.
            ``synapse_models`` (:class:`ParamsTree`)
              Parameter tree, the leaves of which define synapse models.
              Each leaf is used to initialize a :class:`SynapseModel`
              object.
            ``layers`` (:class:`ParamsTree`)
              Parameter tree, the leaves of which define layers. Each leaf
              is used to initialize a :class:``Layer`` or
              :class:``InputLayer`` object depending on the value of their
              ``type`` parameter.
            ``projection_models`` (:class:`ParamsTree`)
              Parameter tree, the leaves of which define projection models.
              Each leaf is used to initialize a :class:``ProjectionModel``
              object.
            ``recorder_models`` (:class:`ParamsTree`)
              Parameter tree, the leaves of which define recorder models.
              Each leaf is used to initialize a :class:``Model`` object.
            ``topology`` (:class:`ParamsTree`)
              :class:`ParamsTree` object without children, the ``params``
              of which may contain a ``projections`` key specifying all the
              individual population-to-population projections within the
              network as a list. :class:`Projection` objects are created
              from the ``topology`` parameters by the
              :func:`Network.build_projections` method. Refer to this
              method for a description of the ``topology`` parameter.
            ``recorders`` (:class:`ParamsTree`)
              :class:``ParamsTree`` object without children, the ``params``
              of which may contain a ``population_recorders`` and a
              ``projection_recorders`` key specifying all the network
              recorders. :class:`PopulationRecorder` and
              :class:`ProjectionRecorder` objects are created from the
              ``recorders`` parameters by the
              :func:`Network.build_recorders` method. Refer to this method
              for a description of the ``recorders`` parameter.
    """

    MANDATORY_CHILDREN = []
    OPTIONAL_CHILDREN = [
        'neuron_models', 'synapse_models', 'layers', 'projection_models',
        'topology', 'recorder_models', 'recorders'
    ]

    def __init__(self, tree=None):
        """Initialize the network object without creating it in NEST."""

        if tree is None:
            tree = ParamsTree({})

        self._created = False
        self._changed = False
        self.tree = tree.copy()

        # Validate tree
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # Check that the "network" tree's params and nest_params keys are empty
        validation.validate(
            "network", dict(tree.params), param_type='params', mandatory=[],
            optional={})
        validation.validate(
            "network", dict(tree.nest_params), param_type='nest_params',
            mandatory=[], optional={})
        # Check that the "network" tree has the correct children and add
        # default children
        validation.validate_children(
            self.tree, mandatory_children=self.MANDATORY_CHILDREN,
            optional_children=self.OPTIONAL_CHILDREN,
        )

        # Build network components
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.neuron_models = {}
        self.synapse_models = {}
        self.recorder_models = {}
        self.layers = {}
        self.projection_models = {}
        self.projections = []
        self.population_recorders = []
        self.projection_recorders = []

        self.build_neuron_models(self.tree.children['neuron_models'])
        self.build_synapse_models(self.tree.children['synapse_models'])
        self.build_recorder_models(self.tree.children['recorder_models'])
        self.build_layers(self.tree.children['layers'])
        self.build_projection_models(self.tree.children['projection_models'])
        # Projections must be built after layers and projection models
        self.build_projections(self.tree.children['topology'])
        # Initialize population recorders and projection recorders
        self.build_recorders(self.tree.children['recorders'])

    @staticmethod
    def build_named_leaves_dict(constructor, node):
        """Construct all leaves and return as a dictionary."""
        named_leaves = {
            name: constructor(name, dict(leaf.params), dict(leaf.nest_params))
            for name, leaf in node.named_leaves(root=False)
        }
        msg = f"Build N={len(named_leaves)} ``{constructor.__name__}`` objects"
        log.info(msg)
        return named_leaves

    def _update_tree_child(self, child_name, tree):
        """Add a child to ``self.tree``"""
        # Convert to ParamsTree and specify parent tree to preserve inheritance
        if not isinstance(tree, ParamsTree):
            child_tree = ParamsTree(tree, parent=self.tree, name=child_name)
        else:
            child_tree = ParamsTree(
                tree.asdict(),
                parent=self.tree,
                name=child_name
            )
        # Add as child
        self.tree.children[child_name] = child_tree

    def build_neuron_models(self, tree):
        """Initialize ``self.neuron_models`` from the leaves of a tree.

        .. note::
            Overwrites the ``'neuron_models'`` in the network's parameters.

        Args:
            tree (tree-like or ParamsTree). Parameter tree, the leaves of
                which define neuron models. Each leaf is used to initialize a
                :class:`Model` object.
        """
        self._update_tree_child('neuron_models', tree)
        self.neuron_models = self.build_named_leaves_dict(
            Model,
            self.tree.children['neuron_models']
        )

    def build_synapse_models(self, tree):
        """Initialize ``self.synapse_models`` from the leaves of a tree.

        .. note::
            Overwrites the ``'synapse_models'`` in the network's parameters.

        Args:
            tree (tree-like or ParamsTree). Parameter tree, the leaves of
                which define neuron models. Each leaf is used to initialize a
                :class:`SynapseModel` object.
        """
        self._update_tree_child('synapse_models', tree)
        self.synapse_models = self.build_named_leaves_dict(
            SynapseModel,
            self.tree.children['synapse_models']
        )

    def build_recorder_models(self, tree):
        """Initialize ``self.recorder_models`` from the leaves of a tree.

        .. note::
            Overwrites the ``'recorder_models'`` in the network's parameters.

        Args:
            tree (tree-like or ``ParamsTree``). Parameter tree, the leaves of
                which define neuron models. Each leaf is used to initialize a
                :class:`Model` object.
        """
        self._update_tree_child('recorder_models', tree)
        self.recorder_models = self.build_named_leaves_dict(
            Model,
            self.tree.children['recorder_models']
        )

    def build_layers(self, tree):
        """Initialize ``self.layers`` from the leaves of a tree.

        .. note::
            Overwrites the ``'layers'`` in the network's parameters.

        Args:
            tree (tree-like or ParamsTree). Parameter tree, the leaves of
                which define layers. Each leaf is used to initialize a
                :class:`Layer` or :class:`InputLayer` objecs depending on the
                value of the ``type`` parameter.
        """
        self._update_tree_child('layers', tree)
        self.layers = {
            name: LAYER_TYPES[leaf.params.get('type', None)](
                name, dict(leaf.params), dict(leaf.nest_params)
            )
            for name, leaf
            in self.tree.children['layers'].named_leaves(root=False)
        }
        log.info(
            f"Build N={len(self.layers)} ``Layer`` or ``InputLayer`` objects."
        )

    def build_projection_models(self, tree):
        """Initialize ``self.projection_models`` from the leaves of a tree.

        .. note::
            Overwrites the ``'projection_models'`` in the network's parameters.

        Args:
            tree (tree-like or ParamsTree). Parameter tree, the leaves of
                which define projection models. Each leaf is used to
                initialize a :class:`ProjectionModel` object.
        """
        self._update_tree_child('projection_models', tree)
        self.projection_models = self.build_named_leaves_dict(
            ProjectionModel,
            self.tree.children['projection_models']
        )

    def build_projections(self, topology_tree):
        """Initialize ``self.projections`` from the ``topology`` tree.

        Initialize ``self.projections`` with a list of :class:`Projection` objects.

        Args:
            topology_tree (tree-like or ParamsTree):
                Tree-like or :class:`ParamsTree` without children, the
                parameters of which may contain a ``projections`` parameter
                entry. (Default: ``[]``). The value of the ``projections``
                parameter is a list of items describing the projections to be
                created. Each item must be a dictionary of the following
                form::

                    {
                        'projection_model' : <projection_model>,
                        'source_layers': <source_layers_list>,
                        'source_population': <source_population>,
                        'target_layers': <target_layers_list>,
                        'target_population': <target_population>,
                    }

                where:

                - ``<projection_model>`` is the name of the projection model.
                  Projection models are specified in the
                  ``projection_models`` network parameter.
                - ``<source_layers_list>`` and ``<target_layers_list>`` are
                  lists of source and target layer names. Projections are
                  created for all (source_layer, target layer) pairs.
                - ``<source_population>`` and ``<target_population>`` are
                  ``None`` or the name of source and target populations for
                  the created projection. If ``None``, all populations in the
                  source or target layer are connected.

                Together, ``<projection_model_name>``,
                ``<source_layer_name>``, ``<source_population_name>``,
                ``<target_layer_name>``, and ``<target_population_name>``
                fully specify each individual projection and must be unique.
        """

        self._update_tree_child('topology', topology_tree)
        topology_tree = self.tree.children['topology']

        OPTIONAL_TOPOLOGY_PARAMS = {
            'projections': []
        }

        # Validate ``topology`` parameter
        validation.validate_children(
            topology_tree, [], []
        )  # No children
        validation.validate(
            'topology', dict(topology_tree.nest_params),
            param_type='nest_params', mandatory=[], optional={}
        )  # No nest_params
        # Only a 'projections' `params` entry
        projection_items = validation.validate(
            'topology', dict(topology_tree.params), param_type='params',
            mandatory=[], optional=OPTIONAL_TOPOLOGY_PARAMS
        )['projections']

        # Get all unique ``(projection_model, source_layer, source_population,
        # target_layer, target_population)`` tuples
        projection_args = self._parse_projection_params(projection_items)

        # Verbose
        c_types_str = ' or '.join([
            c.__name__ for c in CONNECTION_TYPES.values()
        ])
        msg = f"Build N={len(projection_args)} ``{c_types_str}`` objects"
        log.info(msg)

        # Build Projection objects
        projections = []
        for (proj_model, src_lyr, src_pop, tgt_lyr, tgt_pop) in projection_args:
            model = self.projection_models[proj_model]
            source = self.layers[src_lyr]
            target = self.layers[tgt_lyr]
            projections.append(
                CONNECTION_TYPES[model.type](
                    model,
                    source, src_pop,
                    target, tgt_pop
                )
            )

        # Initialize attribute
        self.projections = projections

    def _parse_projection_params(self, projection_items):
        """Return list of tuples specifying all unique projections

        Args:
            projection_items: Content of the ``projections``
                network/topology parameter. See ``self.build_projections`` for
                detailed description.

        Return:
            list: List of unique (<projection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>) tuples specifying all projections in
                the network.
        """
        projection_args = []
        for projection_item in projection_items:
            for source_layer_name, target_layer_name in itertools.product(
                projection_item['source_layers'],
                projection_item['target_layers']
            ):
                projection_args.append(
                    (
                        projection_item['projection_model'],
                        source_layer_name,
                        projection_item['source_population'],
                        target_layer_name,
                        projection_item['target_population'],
                    )
                )

        # Check that there are no duplicates.
        if not len(set(projection_args)) == len(projection_args):
            raise ParameterError(
                """Duplicate projections specified by `projections` topology
                parameter. (<projection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>) tuples should uniquely specify
                projections."""
            )

        return sorted(set(projection_args))

    def build_recorders(self, recorders_tree):
        """Initialize recorders from tree.

        Validates the ``recorders`` parameter tree and calls
        :meth:`Network.build_population_recorders` and
        :meth:`Network.build_projection_recorders` to initialize the
        :meth:`Network.population_recorders` and
        :meth:`Network.projection_recorders` attributes.

        Args:
            recorders_tree (tree-like or ParamsTree): Tree-like or
                :class:`ParamsTree` object without children nor
                ``nest_params``, the parameters of which may contain a
                ``population_recorders`` (default: ``[]``) and a
                ``projection_recorders`` (default: ``[]``) entry specifying
                the network's recorders. The ``population_recorders`` and
                ``projection_recorders`` entries are passed to
                :meth:`Network.build_population_recorders` and
                :meth:`Network.build_projection_recorders` respectively.

        Returns:
            (list(:class:`PopulationRecorder`), list(:class:`ProjectionRecorder`))
        """

        self._update_tree_child('recorders', recorders_tree)
        recorders_tree = self.tree.children['recorders']

        OPTIONAL_RECORDERS_PARAMS = {
            'population_recorders': [],
            'projection_recorders': [],
        }

        # Validate recorders tree
        validation.validate_children(
            recorders_tree, [], []
        )  # No children
        validation.validate(
            'recorders', dict(recorders_tree.nest_params),
            param_type='nest_params',
            mandatory=[], optional={}
        )  # No nest_params
        # Only a 'population_params' or 'projection_params' `params` entry
        recorders_params = validation.validate(
            'recorders', dict(recorders_tree.params), param_type='params',
            mandatory=[], optional=OPTIONAL_RECORDERS_PARAMS
        )

        self.population_recorders = self._build_population_recorders(
            recorders_params['population_recorders']
        )
        self.projection_recorders = self._build_projection_recorders(
                recorders_params['projection_recorders']
        )

    def _build_projection_recorders(self, projection_recorders_items):
        """Return projection recorders specified by a list of recorder params.

        :class:`ProjectionRecorder`s must be built after :class:`Projection`s.

        Arguments:

            projection_recorders_items (list | None):
                Content of the ``projection_recorders`` network/recorders
                parameter. A list of items describing the projection
                recorders to be created. Each item must be a dictionary of
                the following form::

                    {
                        'model': <recorder_model>
                        'projection_model' : <projection_model>,
                        'source_layers': <source_layers_list>,
                        'source_population': <source_population>,
                        'target_layers': <target_layers_list>,
                        'target_population': <target_population>,
                    }

                Where ``<model>`` is the name of the projection recorder model
                (eg 'weight_recorder'). The other keys fully specify the list
                of population-to-population projections of a certain model
                that a projection recorder is created for. Refer to
                :meth:`Network.build_projections` for a full description of
                how the ``<projection_model>``, ``<source_layers>``,
                ``<source_population>``, ``<target_layers>``,
                ``<target_population>`` keys are interpreted.

        Returns:
            list: List of :class:`ProjectionRecorder` objects.
        """
        if projection_recorders_items is None:
            projection_recorders_items = []
        # Get all unique ``(model, projection_model, source_layer,
        # source_population, target_layer, target_population)`` tuples
        proj_recorder_args = []
        for item in projection_recorders_items:
            item = dict(item)  # TODO Fix this
            model = item.pop('model')
            proj_recorder_args += [
                (model,) + proj_args
                for proj_args in self._parse_projection_params([item])
            ]
        proj_recorder_args = sorted(proj_recorder_args)

        # Check that there are no duplicates.
        if not len(set(proj_recorder_args)) == len(proj_recorder_args):
            raise ParameterError(
                """Duplicate projection recorders specified by
                ``projection_recorders`` network/recorders parameter.
                (<recorder_model>, <projection_model_name>, <source_layer_name>,
                <source_population_name>, <target_layer_name>,
                <target_population_name>) tuples should uniquely specify
                projections and projection recorders."""
            )

        projection_recorders = []
        for (
            model, proj_model, src_layer, src_pop, tgt_layer, tgt_pop
        ) in proj_recorder_args:

            matching_projections = [
                c for c in self.projections
                if (c.model.name == proj_model
                    and c.source.name == src_layer
                    and c.source_population == src_pop
                    and c.target.name == tgt_layer
                    and c.target_population == tgt_pop)
            ]

            # Check that all the projections exist in the network
            if not any(matching_projections):
                raise ParameterError(
                    f"Could not create projection recorder {model} for"
                    f" projection `{proj_model}, {src_layer}, {src_pop},"
                    f" {tgt_layer}, {tgt_pop}`: Projection does not exist in "
                    f"the network."
                )
            # Check (again) that projections are unique
            if len(matching_projections) > 1:
                raise ParameterError("Multiple identical projections")

            projection = matching_projections[0]
            # Create projection recorder
            projection_recorders.append(
                ProjectionRecorder(model, projection)
            )

        # Verbose
        msg = f"Build N={len(projection_recorders)} projection recorders."
        log.info(msg)

        return projection_recorders

    def _build_population_recorders(self, population_recorders_items):
        """Return population recorders specified by a list of recorder parameters.

        Arguments:
            population_recorders_items (list | None):
                Content of the ``population_recorders`` network/recorders
                parameter. A list of items describing the population
                recorders to be created and connected to the network. Each
                item must be a dictionary of the following form::

                    {
                        'model' : <model>,
                        'layers': <layers_list>,
                        'populations': <populations_list>,
                    }

                where:

                - ``<model>`` is the model of a recorder.
                - ``<layers_list>`` is None or a list of layer names. If
                  ``None``, all the layers in the network are considered.
                - ``<populations_list>`` is None or a list of populations. If
                  ``None``, all the populations in each layer of interest
                  are considered. For :class:`InputLayer` layers, only the
                  population of parrot neurons can be recorded.
                For each item in the list, a recorder of ``model`` will be
                created and connected to the population(s) of interest of each
                layer(s) of interest.

        Returns:
            list: List of :class:`PopulationRecorder` objects.
        """
        if population_recorders_items is None:
            population_recorders_items = []
        # Get all (model, layer_name, population_name) tuples
        population_recorders_args = []
        # Iterate on layers x population for each item in list
        for item in population_recorders_items:
            model = item['model']
            layer_names = item['layers']
            # Use all layers if <layers> is None
            if layer_names is None:
                layer_names = self.layers.keys()
            for layer_name in layer_names:
                layer = self.layers[layer_name]
                population_names = item['populations']
                # Use all recordable population in layer if <population> is None
                if population_names is None:
                    population_names = layer.recordable_population_names
                # Otherwise use only the populations specified if they exist in
                # the layer
                for population_name in [
                    p for p in population_names
                    if p in layer.recordable_population_names
                ]:
                    population_recorders_args.append(
                        (model, layer_name, population_name)
                    )

        # Verbose
        msg = f"Build N={len(population_recorders_args)} population recorders."
        log.info(msg)

        # Build the unique population recorder objects
        return [
            PopulationRecorder(
                model,
                layer=self.layers[layer_name],
                population_name=population_name
            )
            for (model, layer_name, population_name)
            in sorted(set(population_recorders_args))
        ]

    def __repr__(self):
        return '{classname}({tree})'.format(
            classname=type(self).__name__, tree=(self.tree))

    def __str__(self):
        return repr(self)

    @staticmethod
    def _create_all(objects):
        for obj in tqdm(objects):
            obj.create()

    def _layer_call(self, method_name, *args, layer_type=None, **kwargs):
        """Call a method on each layer."""
        for layer in self._get_layers(layer_type=layer_type):
            method = getattr(layer, method_name)
            method(*args, **kwargs)

    def _layer_get(self, attr_name, layer_type=None):
        """Get an attribute from each layer."""
        return tuple(
            getattr(layer, attr_name)
            for layer in self._get_layers(layer_type=layer_type))

    def _get_layers(self, layer_type=None):
        """Return all layers of a certain type ('Layer' or 'InputLayer')"""
        if layer_type is None:
            return sorted(self.layers.values())
        return [
            l for l in sorted(self.layers.values())
            if type(l).__name__ == layer_type
        ]

    def _recorder_call(self, method_name, *args, recorder_class=None,
                      recorder_type=None, **kwargs):
        """Call a method on all recorder objects

        Args:
            method_name (str): Name of method of recorder objects.
            recorder_class, recorder_type (str or None): Passed to
                :func:`self.get_recorders()`.
            *args: Passed to method ``method_name``.

        Keyword Args:
            **Keyword Args: Passed to method ``method_name``.
        """
        for recorder in self.get_recorders(
            recorder_class=recorder_class,
            recorder_type=recorder_type
        ):
            method = getattr(recorder, method_name)
            method(*args, **kwargs)

    def get_recorders(self, recorder_class=None, recorder_type=None):
        """Yield all :class:`PopulationRecorder` and :class:`ProjectionRecorder` objects.

        Args:
            recorder_class (str or None): Class of queried recorders.
                ``"PopulationRecorder"``, ``"ProjectionRecorder"`` or ``None``.
            recorder_type (str or None): Type of queried recorders.
                ``'multimeter'``, ``'spike_detector'`` or ``'projection_recorder'``.
        """
        if recorder_type in ['multimeter', 'spike_detector']:
            recorder_class = 'PopulationRecorder'
        elif recorder_type in ['weight_recorder']:
            recorder_class = 'ProjectionRecorder'
        elif recorder_type is not None:
            raise ValueError('Recorder type not recognized')
        if recorder_class == 'PopulationRecorder' or recorder_class is None:
            yield from self.get_population_recorders(
                recorder_type=recorder_type
            )
        if recorder_class == 'ProjectionRecorder' or recorder_class is None:
            yield from self.get_projection_recorders(
                recorder_type=recorder_type
            )

    def get_population_recorders(self, recorder_type=None):
        """Yield :class:`PopulationRecorder` objects of type ``recorder_type``."""
        if recorder_type not in [
            "multimeter", "spike_detector", None
        ]:
            raise ValueError('Recorder type not recognized')
        return iter([
            poprec for poprec in self.population_recorders
            if recorder_type is None or poprec.type == recorder_type
        ])

    def get_projection_recorders(self, recorder_type=None):
        """Yield :class:`ProjectionRecorder` objects of type ``recorder_type``."""
        if recorder_type not in [
            "weight_recorder", None
        ]:
            raise ValueError('Unrecognized recorder type')
        yield from self.projection_recorders

    def _get_synapses(self, synapse_type=None):
        """Return synapse models."""
        if synapse_type is None:
            return sorted(self.synapse_models.values())
        return [
            syn for syn in sorted(self.synapse_models.values())
            if syn.type == synapse_type
        ]

    @if_not_created
    def create(self):
        # TODO: use progress bar from PyPhi?
        log.info('Creating neuron models...')
        self._create_all(self.neuron_models.values())
        log.info('Creating synapse models...')
        self._create_all(self.synapse_models.values())
        log.info('Creating recorder models...')
        self._create_all(self.recorder_models.values())
        log.info('Creating layers...')
        self._create_all(self._get_layers())
        log.info('Creating population recorders...')
        self._create_all(self.population_recorders)
        log.info('Creating projection recorders...')
        # ProjectionRecorders must be created BEFORE Projections
        self._create_all(self.projection_recorders)
        log.info('Connecting layers...')
        self._create_all(self.projections)
        self.print_network_size()

    @staticmethod
    def change_synapse_states(synapse_changes):
        """Change parameters for some projections of a population.

        Args:
            synapse_changes (list):
                List of dictionaries each of the form::

                    {
                        'synapse_model': <synapse_model>,
                        'params': {<param1>: <value1>}
                    }

                where the dictionary in ``params`` is passed to
                ``nest.SetStatus()`` to set the parameters for all
                projections with synapse model ``<synapse_model>``.
        """
        import nest
        for changes in tqdm(
                sorted(synapse_changes, key=_synapse_sorting_map),
                desc="-> Changing synapses's state."):
            target_conns = nest.GetConnections(
                synapse_model=changes['synapse_model']
            )
            change_params = changes['params']
            log.info("Changing status for %s projections of type %s. Applying dict: %s", len(target_conns), changes['synapse_model'], change_params)
            nest.SetStatus(target_conns, change_params)

    def set_state(self, unit_changes=None, synapse_changes=None,
                  input_dir=None):
        """Set the state of some units and synapses.

        Args:
            unit_changes (list):
                List of dictionaries specifying the changes applied to the
                networks units::

                    {
                        'layers': <layer_name_list>,
                        'population': <pop_name>,
                        'change_type': <change_type>,
                        'from_array': <from_array>,
                        'nest_params': {
                            <param_name>: <param_change>,
                        },
                    }

                where ``<layer_name_list>`` and ``<population_name>`` specify
                all the individual populations to which the changes are applied:

                - ``<layer_name_list>`` (list(str) | None) is the list of
                  names of the considered layers. If ``None``, the changes
                  may be applied to populations from all the layers.
                  (Default: ``[]``)
                - ``<population_name>`` (str | None) is the name of the
                  considered population in each layer. If ``None``, changes
                  are applied to all the populations of each considered
                  layer. (Default: ``None``)
                and ``<change_type>``, ``<from_array>`` and ``'nest_params'``
                specify the changes applied to units from each of those
                populations:

                - ``<change_type>`` ('constant', 'multiplicative' or
                  'additive'). If 'multiplicative' (resp. 'additive'), the
                  set value for each unit and parameter is the product (resp.
                  sum) between the preexisting value and the given value. If
                  'constant', the given value for each unit is set without
                  regard for the preexisting value. (Default: ``'constant'``)
                - ``<from_array>`` (bool) specifies how the <param_change>
                  value given for each parameter is interpreted:

                  - If ``True``, ``param_change`` should be a numpy array or
                    the relative path from ``input_dir`` to a numpy array.
                    The given or loaded array should have the same dimension
                    as the considered population, and its values are mapped
                    to the population's units to set the ``<param_name>``
                    parameter.
                  - If ``False``, the value in ``param_change`` is
                    used to set the ``<param_name>`` parameter for all the
                    population's units.

                - ``'nest_params'`` (Default: ``{}``) is the dictionary specifying
                  the parameter changes applied to the population units.
                  Items are the name of the modified NEST parameters
                  (``<param_name>``) and the values set (``<param_change>``).
                  The ``<change_type>`` and ``<from_array>`` parameters
                  specify the interpretation of the ``<param_change>`` value.

        Examples:
            >>> # Load parameter files and create the network object
            >>> import denest
            >>> network = denest.Network(denest.ParamsTree.read('<path_to_parameter_file>'))

            >>> # Instantiate the network in the NEST kernel
            >>> network.create()

            >>> # Set the same spike times for all the units of a population of spike
            ... # generators
            >>> network.set_state({
            ...     'layers': ['input_layer'],
            ...     'population_name': 'spike_generator',
            ...     'change_type': 'constant',
            ...     'from_array': False,
            ...     'nest_params': {'spike_times': [1.0, 2.0, 3.0]}
            ... })

            >>> # Set the voltage from values for multiple 2x2 population of neurons: specify
            ... # the array directly
            >>> voltages = np.array([[-70.0, -65.0], [-70.0, -65.0]])
            >>> network.set_state({
            ...     'layers': ['l1', 'l2'],
            ...     'population_name': None,
            ...     'change_type': 'constant',
            ...     'from_array': True,
            ...     'nest_params': {'V_m': voltages}
            ... })

            >>> # Set the voltage from values for a 2x2 population of neurons: pass
            ... # the path to the array
            >>> np.save(voltages, './voltage_change.npy')
            >>> network.set_state({
            ...     'layers': ['l1'],
            ...     'population_name': 'l1_exc',
            ...     'change_type': 'constant',
            ...     'from_array': True,
            ...     'nest_params': {'V_m': './voltage_change.npy'}
            ... })

            >>> # Multiply the leak potential by 2 for all the units
            >>> network.set_state({
            ...     'layers': ['l1'],
            ...     'population_name': 'l1_exc',
            ...     'change_type': 'multiplicative',
            ...     'from_array': False,
            ...     'nest_params': {'g_peak_AMPA': 2.0}
            ... })
        """
        UNIT_CHANGES_OPTIONAL = {
            'nest_params': {},
            'population_name': None,
            'change_type': 'constant',
            'from_array': False,
            'layers': [],
        }

        if unit_changes is None:
            unit_changes = []

        for changes in sorted(unit_changes, key=_unit_sorting_map):

            changes = validation.validate(
                'Unit changes dictionary',
                changes,
                mandatory=[],
                optional=UNIT_CHANGES_OPTIONAL,
            )

            # Iterate on layers
            layer_names = changes.get('layers', [])
            if layer_names is None:
                layers = self._get_layers()
            else:
                layers = [self.layers[layer_name] for layer_name in layer_names]

            for layer in layers:

                layer.set_state(
                    nest_params=changes['nest_params'],
                    population_name=changes['population_name'],
                    change_type=changes['change_type'],
                    from_array=changes['from_array'],
                    input_dir=input_dir,
                )

    def save_metadata(self, output_dir):
        """Save network metadata.

            - Save recorder metadata
        """
        # Save recorder metadata
        self._recorder_call('save_metadata', output_dir)

    @staticmethod
    def print_network_size():
        import nest
        log.info('Network size (including recorders and parrot neurons):\n'
                 'Number of nodes: %s\n'
                 'Number of projections: %s',
                 nest.GetKernelStatus('network_size'),
                 nest.GetKernelStatus('num_connections')
                 )

    def populations_by_gids(self, layer_type='Layer'):
        """Return a dict of the form ``{'gid': (layer_name, pop_name)}``."""
        all_pops = {}
        for layer in self._get_layers(layer_type=layer_type):
            all_pops.update({
                gid: (layer.name, population)
                for gid, population in layer.populations.items()
            })
        return all_pops


def _unit_sorting_map(unit_change):
    """Map by (layer, population, proportion, params_items for sorting."""
    return (unit_change.get('layers', 'None'),
            unit_change.get('population', 'None'),
            unit_change.get('population', 'None'),
            sorted(unit_change.get('params', {}).keys()))


def _synapse_sorting_map(synapse_change):
    """Map by (synapse_model, params_items) for sorting."""
    return (synapse_change['synapse_model'],
            sorted(synapse_change['params'].items()))
