#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/layers.py

"""Layer objects."""

import itertools
import logging
from pathlib import Path

import numpy as np

from ..base_object import NestObject
from ..utils.validation import ParameterError
from .utils import flatten, if_created, if_not_created

log = logging.getLogger(__name__)


class AbstractLayer(NestObject):
    """Abstract base class for a layer.

    Defines the layer interface.
    """

    def __init__(self, name, params, nest_params):
        super().__init__(name, params, nest_params)
        self._gid = None
        self._gids = None  # list of layer GIDs
        self._layer_locations = {}  # {<gid>: (row, col)}
        self._population_locations = {}  # {<gid>: (row, col, unit_index)}
        self._populations = params["populations"]  # {<population>: <number>}
        self._shape = nest_params["rows"], nest_params["columns"]
        # Record if we change some of the layer units' state probabilistically
        self._prob_changed = False

    #TODO
    def __iter__(self):
        """Iterate on layer locations."""
        yield from itertools.product(range(self.shape[0]), range(self.shape[1]))

    @if_not_created
    def create(self):
        """Create the layer in NEST."""
        raise NotImplementedError

    @property
    def populations(self):
        """Return ``{<population_name>: <number_units>}`` dictionary."""
        return self._populations

    @property
    def layer_shape(self):
        """Return shape of the layer."""
        return self.shape

    @property
    def population_shape(self):
        """Return shape of a population = layer_shape + (pop_n_units,)"""
        return {
            population_name: self.shape + (self.populations[population_name],)
            for population_name in self.population_names
        }

    @property
    @if_created
    def gid(self):
        """Return the NEST global ID (GID) of the layer object."""
        return self._gid

    @if_created
    def _connect(self, target, nest_params):
        """Connect to target layer. Called by `Projection.create()`"""
        # NOTE: Don't use this method directly; use a Projection instead
        from nest import topology as tp

        tp.ConnectLayers(self.gid, target.gid, nest_params)

    def gids(self, population=None, location=None, population_location=None):
        """Return element GIDs, optionally filtered by population/location.

        Args:
            population (str): Matches population name that has ``population`` as
                a substring.
            location (tuple[int]): The location within the layer to filter
                by.
            population_location (tuple[int]): The location within the population
                to filter by.

        Returns:
            list: The GID(s).
        """
        raise NotImplementedError

    def change_unit_states(
        self, changes_dict, population=None, proportion=1.0, change_type="constant"
    ):
        """Set parameters for some units.

        Args:
            changes_dict (dict): Dictionary specifying changes applied to
                selected units, of the form::
                    {
                        <param_1>: <change_value_1>,
                        <param_2>: <change_value_2>
                    }
                The values are set multiplicatively or without modification
                depending on the ``change_type`` parameter.
            population (str | None): Name of population from which we select
                units. All layer's populations if None.
            proportion (float): Proportion of candidate units to which the
                changes are applied. (default 1.0)
            change_type (str): 'constant' (default) or 'multiplicative'.
                Specifies how parameter values are set from ``change_dict``. If
                "constant", the values in ``change_dict`` are set to the
                corresponding parameters without modification. If
                "multiplicative", the values in ``change_dict`` are multiplied
                to the current value of the corresponding parameter for each
                unit.
        """
        if change_type not in ["constant", "multiplicative"]:
            raise ParameterError(
                "``change_type`` argument should 'constant' or 'multiplicative'"
            )
        if proportion > 1 or proportion < 0:
            raise ParameterError("``proportion`` parameter should be within [0, 1]")
        if not changes_dict:
            return
        if self._prob_changed and proportion != 1.0:
            raise Exception(
                "Attempting to change probabilistically some "
                "units' state multiple times."
            )
        all_gids = self.gids(population=population)
        if proportion != 1.0:
            log.info(f"    Select subset of gids (proportion = {proportion})")
        gids_to_change = self.get_gids_subset(all_gids, proportion)
        log.info(
            '    Apply "%s" parameter changes on %s/%s units (layer=%s, population=%s)',
            change_type,
            len(gids_to_change),
            len(all_gids),
            self.name,
            population,
        )
        self.apply_unit_changes(gids_to_change, changes_dict, change_type=change_type)
        self._prob_changed = True

    @if_created
    def set_state(self, nest_params=None, population_name=None,
                  change_type='constant', from_array=False, input_dir=None):
        """Set the state of some of the layer's populations."""

        if input_dir is None:
            input_dir = Path('./')

        # Iterate on populations
        if population_name is None:
            population_names = self.population_names
        else:
            population_names = [population_name]

        for population_name in population_names:
            population_shape = self.population_shape[population_name]

            # For all the considered parameters, unfold the `param_change` value
            # into an array to map to the population
            param_arrays = {}
            for param_name, param_change in nest_params.items():

                # Get array of values the same shape as the population
                # Option 1: map from numpy array directly provided
                if from_array and isinstance(param_change, (np.ndarray)):
                    values_array = param_change
                    from_file = False
                # Option 2: map from numpy array loaded from file
                elif from_array:
                    path = Path(input_dir)/Path(param_change)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Could not load array from file at {path}"
                        )
                    values_array = np.load(path)
                    from_file = True
                # Option 3: Same value applied to all the units in the pop
                else:
                    # This does not work to make an array of lists
                    # values_array = np.full(population_shape, param_change)
                    # This can make array of lists
                    values_array = np.frompyfunc(
                        lambda: param_change, 0, 1)(
                            np.empty(population_shape, dtype=object)
                    )

                # Provided array has correct dimension
                if not sorted(values_array.shape) == sorted(population_shape):
                    raise ValueError(
                        f'Layer `{self.name}`, population `{population_name}`, '
                        f'parameter `{param_name}``, '
                        f'(array from file)={from_file}: '
                        f'Array has '
                        f' incorrect shape. Expected shape `{population_shape}`'
                        f', got shape `{values_array.shape}`'
                    )

                log.info(
                    f"Layer='{self.name}', pop='{population_name}': Applying "
                    f"'{change_type}' change, param='{param_name}', "
                    f"{'from array' if from_array else 'from single value'}')"
                )

                param_arrays[param_name] = values_array

            # Set all the parameters at once for each unit in the population
            for idx, x in np.ndenumerate(values_array):
                tgt_gid = self.gids(
                    population=population_name,
                    population_location=idx,
                )
                self.set_unit_state(
                    tgt_gid,
                    {
                        param_name: param_arrays[param_name][idx]
                        for param_name in param_arrays.keys()
                    },
                    change_type=change_type
                )

    @staticmethod
    def set_unit_state(gids, params, change_type="constant"):
        """Change some units'  parameter in NEST.

        Args:
            gids (list(int)): Gids of units to change the state of
            params (dict): ``{param_name: param_change}`` dictionary describing
                the modified parameters. The `param_change` values used for
                modification are set directly or added/multiplied to the current
                value of the parameter for each unit, depending on the
                ``'change_type'`` kwarg

        Keyword Args:
            change_type ('constant', 'multiplicative' or 'additive'). If
                'multiplicative' (resp. 'additive'), the set value for each unit
                and each parameter is the product (resp. sum) between the
                preexisting value and the given value. If 'constant', the given
                value for each unit is set without regard for the preexisting
                value. (default: 'constant')
        """
        import nest

        CHANGE_TYPES = ["constant", "multiplicative", "additive"]
        if change_type not in CHANGE_TYPES:
            raise ValueError(
                '``change_type`` param should be one of {CHANGE_TYPES}'
            )

        if change_type == "constant":
            nest.SetStatus(gids, params)
        else:
            current_values = {
                param_name: nest.GetStatus(gids, param_name)
                for param_name in params.keys()
            }
            if not all([
                isinstance(v, float)
                for unit_values in current_values.values()
                for v in unit_values
            ]):
                raise ValueError(
                    "Can't set state multiplicatively for non-float"
                    f" parameter(s) {params.keys()}."
                    f" Expecting ``change_type='constant'``."
                )
            if change_type == 'multiplicative':
                set_values = {
                    param_name: [
                        v * params[param_name]
                        for v in current_values[param_name]
                    ]
                    for param_name in params.keys()
                }  # {param: [v_gid1, v_gid2, ...]}
            elif change_type == 'additive':
                set_values = {
                    param_name: [
                        v + params[param_name]
                        for v in current_values[param_name]
                    ]
                    for param_name in params.keys()
                }  # {param: [v_gid1, v_gid2, ...]}
            else:
                assert False
            nest.SetStatus(
                gids,
                [
                    {
                        param_name: set_values[param_name][gid_i]
                        for param_name in params.keys()
                    }  # {param: gid_param_value}
                    for gid_i in range(len(gids))
                ]  # One param dict per unit
            )


class Layer(AbstractLayer):
    """Represents a NEST layer composed of populations of units

    Args:
        name (str): Name of the layer
        params (dict-like): Dictionary of parameters. The following parameters
            are expected:
                populations (dict): Dictionary of the form ``{<model>: <number>}
                    specifying the elements within the layer. Analogous to the
                    ``elements`` nest.Topology parameter
        nest_params (dict-like): Dictionary of parameters that will be passed
            to NEST during the ``nest.CreateLayer`` call. The following
            parameters are mandatory: ``['rows', 'columns']``. The
            ``elements`` parameter is reserved. Please use the ``populations``
            parameter instead to specify layer elements.
    """

    # Validation of `params`
    RESERVED_PARAMS = None
    MANDATORY_PARAMS = ["populations"]
    OPTIONAL_PARAMS = {"type": None}
    # Validation of `nest_params`
    RESERVED_NEST_PARAMS = ["elements"]
    MANDATORY_NEST_PARAMS = ["rows", "columns"]
    OPTIONAL_NEST_PARAMS = None

    def __init__(self, name, params, nest_params):
        super().__init__(name, params, nest_params)
        self.nest_params["elements"] = self._build_elements()

    def _build_elements(self):
        """Convert ``populations`` parameters to format expected by NEST

        From the ``populations`` layer parameter, which is a dict of the
        form::
            {
                <population_name>: <number_of_units>
            }
        return a NEST element specification, which is a list of the form::
            [<model_name>, <number of units>]
        """
        populations = self.params["populations"]
        if not populations or any(
            [not isinstance(n, int) for n in populations.values()]
        ):
            raise ParameterError(
                "Invalid format for `populations` parameter {populations} of "
                f"layer {str(self)}: expects non-empty dictionary of the form"
                "`{<population_name>: <number_of_units>}` with integer values"
            )
        # Map types to numbers
        return flatten(
            [population, number] for population, number in populations.items()
        )

    @if_not_created
    def create(self):
        """Create the layer in NEST and update attributes."""
        from nest import topology as tp
        import nest

        self._gid = tp.CreateLayer(self.nest_params)
        self._gids = nest.GetNodes(self.gid)[0]
        # Update _layer_locations: eg ``{gid: (row, col)}``
        # and _population_locations: ``{gid: (row, col, unit_index)}``
        for index, _ in np.ndenumerate(np.empty(self.shape)):  # Hacky
            for population in self.population_names:
                # Match population and location
                loc_pop_gids = [
                    gid for gid in tp.GetElement(self._gid, index[::-1])
                    if nest.GetStatus((gid,), "model")[0] == population
                ]
                # IMPORTANT: rows and columns are switched in the GetElement
                # query
                for k, gid in enumerate(loc_pop_gids):
                    self._layer_locations[gid] = index
                    self._population_locations[gid] = index + (k,)
        assert set(self._gids) == set(self._layer_locations.keys())

    @if_created
    def gids(self, population=None, location=None, population_location=None):
        import nest

        return [
            gid for gid in self._gids
            if (population is None
                or nest.GetStatus((gid,), "model")[0] == population) \
            and (location is None
                 or list(self.locations[gid]) == list(location)) \
            and (population_location is None
                 or list(self.population_locations[gid]) == list(population_location))
        ]

    @property
    def shape(self):
        """Return layer shape (eg ``(nrows, ncols)``)"""
        return self._shape

    @property
    def layer_shape(self):
        """Return layer shape (eg ``(nrows, ncols)``)"""
        return self.shape

    @property
    def population_shapes(self):
        """Return population shapes: ``{<pop_name>: (nrows, ncols, nunits)``"""
        return {
            pop_name: self.shape + (self.populations[pop_name],)
            for pop_name in self.populations
        }

    @property
    @if_created
    def locations(self):
        """Return ``{<gid>: index}`` dict of locations within the layer."""
        return self._layer_locations

    @property
    @if_created
    def population_locations(self):
        """Return ``{<gid>: index}`` dict of locations within the population.

        There's an extra (last) dimension for the population locations compared
        to the [layer] locations, corresponding to the index of the unit within
        the population.
        """
        return self._population_locations

    @if_created
    @staticmethod
    def position(*args):
        import nest.topology as tp

        return tp.GetPosition(args)

    @property
    def population_names(self):
        """Return a list of population names within this layer."""
        return list(self.params["populations"].keys())

    @property
    def recordable_population_names(self):
        """Return list of names of recordable population names in this layer."""
        return self.population_names


class InputLayer(Layer):
    """A layer of stimulators

    ``InputLayer`` extends the ``Layer`` class to handle layers of stimulation
    devices. `InputLayer` parameters should specify a single population of
    stimulation devices.

    If the `add_parrot` layer parameter is True (default True), a second
    population of parrot neurons, with the same number of units, will be created
    and connected one-to-one to the population of stimulators, to allow
    recording of activity in the layer.
    """

    OPTIONAL_PARAMS = {"type": 'InputLayer', 'add_parrots': True}

    # Append ``Layer`` docstring
    __doc__ += "\n".join(Layer.__doc__.split("\n")[1:])

    PARROT_MODEL = "parrot_neuron"

    def __init__(self, name, params, nest_params):

        # TODO make deepcopies everywhere
        import copy

        params = copy.deepcopy(params)

        # Check populations and add a population of parrot neurons
        populations = params["populations"]
        if len(populations) != 1:
            raise ParameterError(
                f"Invalid `population` parameter for `InputLayer` layer {name}."
                f" InputLayers should be composed of a single population of"
                f"stimulation devices."
                f" Please check the `population` parameter: {populations}"
            )
        # Save the stimulator type
        stimulator_model, nunits = list(populations.items())[0]
        self.stimulator_model = stimulator_model
        self.stimulator_type = None  # TODO: Check stimulator type
        # Add a parrot population entry
        if 'add_parrot' not in params:
            params['add_parrots'] = True
        if params['add_parrots']:
            populations[self.PARROT_MODEL] = nunits
        params["populations"] = populations

        # Initialize the layer
        super().__init__(name, params, nest_params)

    def create(self):
        """Create the layer and connect the stimulator and parrot populations"""
        super().create()
        import nest

        # Connect stimulators to parrots, one-to-one
        stim_gids = []
        parrot_gids = []
        for loc in self:
            stim_gids += self.gids(location=loc, population=self.stimulator_model)
            parrot_gids += self.gids(location=loc, population=self.PARROT_MODEL)
        nest.Connect(stim_gids, parrot_gids, "one_to_one", {"model": "static_synapse"})
        # Get stimulator type
        self.stimulator_type = nest.GetDefaults(self.stimulator_model, "type_id")

    @property
    def recordable_population_names(self):
        """Return list of names of recordable population names in this layer."""
        return ["parrot_neuron"]
