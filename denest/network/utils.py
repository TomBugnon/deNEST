#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/utils.py

"""Utilities for building network objects."""

import functools
import logging

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def flatten(seq):
    """Flatten an iterable of iterables into a tuple."""
    return tuple(item for subseq in seq for item in subseq)


def indent(string, amount=2):
    """Indent a string by an amount."""
    return "\n".join((" " * amount) + line for line in string.split("\n"))


class NotCreatedError(AttributeError):
    """Raised when a ``NestObject`` needs to have been created, but wasn't."""

    pass


def if_not_created(method):
    """Only call a method if the ``_created`` attribute isn't set.

    After calling, sets ``_created = True``.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-docstring
        if self._created:
            log.warning(
                "Attempted to create object more than once:\n%s", indent(str(self))
            )
            return
        try:
            self._created = True
            value = method(self, *args, **kwargs)
        except Exception as error:
            self._created = False
            raise error
        return value

    return wrapper


def if_created(method):
    """Raise an error if the `_created` attribute is not set."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-docstring
        if not self._created:
            raise NotCreatedError("Must call `create()` first:\n" + indent(str(self)))
        return method(self, *args, **kwargs)

    return wrapper
