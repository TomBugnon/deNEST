#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/validation.py

"""Validation and update of parameters dictionaries."""

import copy as cp


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


def validate_children(name, children_list, mandatory_children=None):
    """Validate a `Params` node children.

    Args:
        name (str): Name of the `Params` node we're validating

    Kwargs:
        children_list (list(str) | None): ``None`` or the list of names of
            children nodes that are expected in the validated node. Ignored if
            ``None``
    """

    if mandatory_children is None:
        return

    error_msg_base = (
        f"Invalid set of children for ``Params`` node ``{name}``:\n"
    )

    missing_children = list(set(mandatory_children) - set(children_list))
    if any(missing_children):
        raise MissingChildNodeError((
            error_msg_base +
            f"The following `Params` children subtrees are missing: "
            f"{missing_children}"
        ))

    extra_children = list(set(children_list) - set(mandatory_children))
    if any(extra_children):
        raise UnrecognizedChildNodeError((
            error_msg_base +
            f"The following `Params` children subtrees are unexpected: "
            f"{extra_children}"
        ))
    return


def validate(name, params, param_type='params', reserved=None,
             mandatory=None, optional=None):
    """Validate and set default values for parameter dictionary.

    Args:
        name (str): Name of the object we're validating
        params (dict-like): Parameter dictionary to validate.

    Kwargs:
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
            print(f"Object {name}, {param_type}: Set default value for optional"
                  f"parameters: {missing_optional}")
            params = cp.deepcopy(params)
            params.update(missing_optional)

    return params
