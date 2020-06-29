#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/validation.py

"""Validation and update of parameters dictionaries."""

import copy as cp
import logging
from pprint import pformat

from ..parameters import ParamsTree

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ParameterError(ValueError):
    """Raised when there is an error with parameters"""
    pass


class ReservedParameterError(ParameterError):
    """Raised when parameter dictionary contains a reserved parameter."""
    pass


class MissingParameterError(ParameterError):
    """Raised when a mandatory parameter is missing."""
    pass


class UnrecognizedParameterError(ParameterError):
    """Raised when a parameter is not recognized."""
    pass


class MissingChildNodeError(ValueError):
    """Raised when a child `Params` node is missing."""
    pass


class UnrecognizedChildNodeError(ValueError):
    """Raised when a child `Params` node is not recognized."""
    pass


# TODO Add as ParamsTree method
def validate_children(tree, mandatory_children=None, optional_children=None):
    """Validate a `ParamsTree` node children.

    Args:
        tree (ParamsTree): The tree we're validating. Missing children are added
            in place as empty ParamsTree nodes.

    Keyword Args:
        mandatory_children (list(str) | None): ``None`` or the list of names of
            children nodes that are expected in the tree. Ignored if
            ``None``
        optional_children (list(str) | None): ``None`` or the list of names of
            children nodes that are optional in the tree. Ignored if
            ``None``. If missing, empty children ParamsTree are added in place.
    """

    if mandatory_children is None:
        mandatory_children = []
    if optional_children is None:
        optional_children = []

    children_list = tree.children.keys()
    error_msg_base = (
        f"Invalid set of children for ``ParamsTree`` node ``{tree}``:\n"
    )

    missing = list(set(mandatory_children) - set(children_list))
    if any(missing):
        raise MissingChildNodeError((
            error_msg_base +
            f"The following children subtrees are missing: "
            f"{missing}"
        ))

    extra = list(
        set(children_list) - set(mandatory_children).union(optional_children)
    )
    if any(extra):
        raise UnrecognizedChildNodeError((
            error_msg_base +
            f"The following `ParamsTree` children subtrees are unexpected: "
            f"{extra}"
        ))

    for optional in optional_children:
        if optional is not None:
            if optional not in tree.children:
                log.info("'%s' tree: adding empty child %s", tree.name, optional)
                tree.children[optional] = ParamsTree(
                    {}, parent=tree, name=optional
                )


def validate(name, params, param_type='params', reserved=None,
             mandatory=None, optional=None):
    """Validate and set default values for parameter dictionary.

    Args:
        name (str): Name of the object we're validating
        params (dict-like): Parameter dictionary to validate.

    Keyword Args:
        param_type (str): Type of parameter dictionary we're validating.
            'params' or 'nest_params'
        reserved (list(str) | None): List of "reserved" parameters that
            shouldn't be present in ``params``. Ignored if None.
        mandatory (list(str) | None): List of parameters that are
            expected in the dictionary. Ignored if ``None``.
        optional (dict | None): Dict of parameters that are
            recognized but optional, along with their default value. Ignored if
            ``None``. If not None, the list of recognized parameters is
            ``mandatory + optional`` and we throw an error if one
            of the params is not recognized. Optional parameters missing from
            `params` are added along with their default value.
    Returns:
        params: The parameter dictionary updated with default values.
    """

    assert param_type in ['params', 'nest_params']

    error_msg_base = (
        f"Invalid parameter for object `{name}` in `{param_type}` parameter "
        f"dictionary: \n`{params}`.\n"
    )

    # Check that there are no forbidden parameters
    if (
        reserved is not None
        and any([key in reserved for key in params.keys()])
    ):
        raise ReservedParameterError(
            error_msg_base + (
                f"The following parameters are reserved: {reserved}"
            )
        )

    # Check that mandatory params are here
    if (
        mandatory is not None
        and any([key for key in mandatory if key not in params.keys()])
    ):
        raise MissingParameterError(
            error_msg_base + (
                f"The following parameters are mandatory: {mandatory}"
            )
        )

    # Check recognized parameters
    if optional is not None:
        assert mandatory is not None
        recognized_params = list(optional.keys()) + mandatory
        if any([key not in recognized_params for key in params.keys()]):
            raise UnrecognizedParameterError(
                error_msg_base + (
                    f"The following parameters are recognized:\n"
                    f"{mandatory} (mandatory)\n"
                    f"{list(optional.keys())} (optional)"
                )
            )

        # Add default values:
        missing_optional = {
            k: v
            for k, v in optional.items()
            if k not in params.keys()
        }
        if any(missing_optional):
            log.info("Object `%s`: %s: using default value for optional parameters:\n%s",
                     name, param_type, pformat(missing_optional))
            params = cp.deepcopy(params)
            params.update(missing_optional)

    return params
